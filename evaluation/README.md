# Reproducible Paper 2 Evaluation

Run the full configured KITTI drive with:

```powershell
.\venv\Scripts\python.exe evaluation\run_experiments.py
```

Outputs are written to `results/`. The KITTI Raw source used by this project
does not provide a separate ground-truth localization stream; therefore
GPS-to-filter distances are labeled as agreement/deviation proxies.
