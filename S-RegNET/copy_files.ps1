$src = "c:\Users\mahit\OneDrive\Desktop\Study\Keyreg\S-RegNET"
$dst = "c:\Users\mahit\OneDrive\Desktop\Study\Research\S-RegNET"

Copy-Item -Path "$src\model.py" -Destination "$dst\model.py" -Force
Copy-Item -Path "$src\losses.py" -Destination "$dst\losses.py" -Force
Copy-Item -Path "$src\train.py" -Destination "$dst\train.py" -Force
Copy-Item -Path "$src\inference.py" -Destination "$dst\inference.py" -Force
Copy-Item -Path "$src\wm_template.py" -Destination "$dst\wm_template.py" -Force
Copy-Item -Path "$src\experiment_routing.py" -Destination "$dst\experiment_routing.py" -Force
Copy-Item -Path "$src\label_mapping.py" -Destination "$dst\label_mapping.py" -Force

if (Test-Path "$dst\eval") { Remove-Item -Path "$dst\eval" -Recurse -Force }
Copy-Item -Path "$src\eval" -Destination "$dst\eval" -Recurse -Force

if (Test-Path "$dst\utils") { Remove-Item -Path "$dst\utils" -Recurse -Force }
Copy-Item -Path "$src\utils" -Destination "$dst\utils" -Recurse -Force

Write-Host "Copy completed."
