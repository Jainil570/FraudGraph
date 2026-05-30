$files = @(
    'c:\programs\projects\ml\FraudGraph\src\utils\data_loader.py',
    'c:\programs\projects\ml\FraudGraph\src\training\trainer.py',
    'c:\programs\projects\ml\FraudGraph\src\utils\metrics.py',
    'c:\programs\projects\ml\FraudGraph\src\evaluation\evaluator.py',
    'c:\programs\projects\ml\FraudGraph\src\tracking\mlflow_tracker.py',
    'c:\programs\projects\ml\FraudGraph\src\models\gcn.py',
    'c:\programs\projects\ml\FraudGraph\src\models\gat.py',
    'c:\programs\projects\ml\FraudGraph\src\utils\visualization.py',
    'c:\programs\projects\ml\FraudGraph\notebooks\01_eda.py',
    'c:\programs\projects\ml\FraudGraph\notebooks\02_baseline.py',
    'c:\programs\projects\ml\FraudGraph\notebooks\03_gcn_gat.py'
)

foreach ($path in $files) {
    if (Test-Path $path) {
        $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
        $content = $content -replace [char]0x2192, '->'
        $content = $content -replace [char]0x2190, '<-'
        $content = $content -replace [char]0x2605, '*'
        $content = $content -replace [char]0x2502, '|'
        $content = $content -replace [char]0x2500, '-'
        $content = $content -replace [char]0x2014, '--'
        $content = $content -replace [char]0x2013, '-'
        [System.IO.File]::WriteAllText($path, $content, [System.Text.Encoding]::UTF8)
        Write-Host "Fixed: $path"
    } else {
        Write-Host "Not found: $path"
    }
}
Write-Host "Done."
