"""
Batched neural network inference for MCTS self-play.

Runs multiple games concurrently in threads. When MCTS needs a NN
evaluation, the request is queued and batched with other pending
requests for a single GPU forward pass. While the GPU processes one
batch, CPU threads continue preparing more evaluations (pipelining).

Usage:
    server = BatchedInferenceServer(net, HiveTokenBatch.collate, device)
    server.start()
    batched_net = BatchedPredictor(server)
    # Pass batched_net to MCTS instead of the real net
    mcts = MCTS(batched_net, encoder, config)
    ...
    server.stop()

Key tuning parameter: max_wait_ms
    After the first item arrives, the server waits up to max_wait_ms for
    more items before firing the batch. This should be close to the time
    it takes for all workers to finish their CPU work and submit.
    Rule of thumb: num_workers * ~2ms (CPU overhead per MCTS sim).
    With 16 workers: ~30ms. With 32 workers: ~60ms.
    Too short → tiny batches, low GPU utilization.
    Too long → unnecessary latency with few workers.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F


class _InferenceRequest:
    """A single pending inference request from a game worker thread."""

    __slots__ = ("state", "mask", "event", "result_probs", "result_value")

    def __init__(self, state, mask: np.ndarray) -> None:
        self.state = state  # encoded state (HiveTokenSequence or HiveGraph)
        self.mask = mask  # legal action mask, shape (ACTION_SPACE,)
        self.event = threading.Event()
        self.result_probs: np.ndarray | None = None
        self.result_value: float = 0.0


class BatchedInferenceServer:
    """
    Collects NN inference requests from multiple threads and processes
    them in batches on the GPU.

    Args:
        net: The neural network (HiveTransformer or HiveGNN).
        collate_fn: Callable that takes list[state] and returns a batch
            object with a .to(device) method (e.g. HiveTokenBatch.collate).
        device: torch.device for inference.
        max_batch_size: Maximum states per forward pass.
        max_wait_ms: After the first item arrives, wait this many ms for
            more before firing the batch. Should be ≈ num_workers × 2ms.
            Default 10ms is a safe starting point; increase toward
            num_workers × 2ms for higher throughput (e.g. 30ms for 16 workers).
    """

    def __init__(
        self,
        net,
        collate_fn,
        device: torch.device,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ) -> None:
        self.net = net
        self.collate_fn = collate_fn
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: collections.deque[_InferenceRequest] = collections.deque()
        self._lock = threading.Lock()
        self._has_items = threading.Condition(self._lock)
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._running = False

        # Batch size statistics (for tuning max_wait_ms)
        self._total_batches: int = 0
        self._total_items: int = 0

    def start(self) -> None:
        """Start the batch processing thread."""
        self.net.eval()
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Signal the processing thread to stop and wait for it."""
        self._running = False
        with self._has_items:
            self._has_items.notify_all()
        self._thread.join(timeout=5.0)

        # Print batch size statistics
        if self._total_batches > 0:
            avg = self._total_items / self._total_batches
            print(
                f"  [BatchedInference] {self._total_batches} batches, "
                f"avg size {avg:.1f}/{self.max_batch_size} "
                f"(wait={self.max_wait_ms:.0f}ms)"
            )

    def submit(
        self, encoded_state, legal_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Submit an inference request. Blocks until the result is ready.

        Args:
            encoded_state: Encoded game state (HiveTokenSequence or HiveGraph).
            legal_mask: Legal action mask, shape (ACTION_SPACE,), float32.

        Returns:
            (action_probs, value) matching the net.predict() interface.
        """
        req = _InferenceRequest(encoded_state, legal_mask)
        with self._has_items:
            self._queue.append(req)
            self._has_items.notify()
        req.event.wait()
        if req.result_probs is None:
            raise RuntimeError("BatchedInferenceServer crashed; see traceback above.")
        return req.result_probs, req.result_value

    def _process_loop(self) -> None:
        """Main loop: collect batches and run inference."""
        try:
            while self._running:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
        except Exception as e:
            import traceback
            print(f"\n[BatchedInference] Server thread crashed: {e}", flush=True)
            traceback.print_exc()
            # Unblock all waiting workers so they raise an error instead of hanging
            self._running = False
            with self._lock:
                pending = list(self._queue)
                self._queue.clear()
            for req in pending:
                req.result_probs = None
                req.event.set()

    def _collect_batch(self) -> list[_InferenceRequest]:
        """
        Collect up to max_batch_size requests.

        Strategy: wait for the first item, then wait up to max_wait_ms
        for more items. This allows all worker threads to finish their
        CPU work and submit before the batch fires.
        """
        with self._has_items:
            # Wait for at least one item
            while not self._queue and self._running:
                self._has_items.wait(timeout=0.05)

            if not self._queue:
                return []

            # Got the first item — now wait for more up to max_wait_ms
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(self._queue) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._has_items.wait(timeout=remaining)

            # Drain queue up to max_batch_size
            batch: list[_InferenceRequest] = []
            while self._queue and len(batch) < self.max_batch_size:
                batch.append(self._queue.popleft())

            return batch

    def _process_batch(self, batch: list[_InferenceRequest]) -> None:
        """Run batched forward pass and distribute results."""
        self._total_batches += 1
        self._total_items += len(batch)

        states = [req.state for req in batch]
        masks_np = [req.mask for req in batch]

        with torch.no_grad():
            collated = self.collate_fn(states).to(self.device)
            masks = torch.stack(
                [torch.from_numpy(m) for m in masks_np]
            ).to(self.device)

            policy_logits, values, _aux = self.net.forward(collated)

            # Apply legal mask and softmax
            policy_logits = policy_logits.masked_fill(
                masks == 0, float("-inf")
            )
            action_probs = F.softmax(policy_logits, dim=-1)

            # Move to CPU
            action_probs_np = action_probs.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

        # Distribute results and unblock waiting threads
        for i, req in enumerate(batch):
            req.result_probs = action_probs_np[i]
            req.result_value = float(values_np[i])
            req.event.set()


class BatchedPredictor:
    """
    Drop-in replacement for a neural network during MCTS.

    Has the same predict(state, mask) interface that MCTS expects,
    but internally submits requests to a BatchedInferenceServer
    for batched processing.
    """

    def __init__(self, server: BatchedInferenceServer) -> None:
        self.server = server

    def predict(
        self, encoded_state, legal_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Predict action probabilities and value for a single state.

        Blocks until the server processes this request as part of a batch.
        """
        return self.server.submit(encoded_state, legal_mask)
