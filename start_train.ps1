$env:PYTHONUNBUFFERED = '1'
Set-Location 'C:\Users\danek\hive_claude\hive_neuralnet'
Start-Process `
    -FilePath '.venv\Scripts\python.exe' `
    -ArgumentList '-u','-m','hive_gnn.train',
        '--resume','checkpoints_gnn/hive_gnn_checkpoint_0038.pt',
        '--iterations','15',
        '--games','100',
        '--simulations','100',
        '--workers','96',
        '--inference-batch','96',
        '--inference-wait-ms','5',
        '--queen-pressure-games','25' `
    -RedirectStandardOutput 'train_gnn.log' `
    -RedirectStandardError 'train_gnn_err.log' `
    -NoNewWindow `
    -WorkingDirectory 'C:\Users\danek\hive_claude\hive_neuralnet'
Write-Host 'Training process started'
