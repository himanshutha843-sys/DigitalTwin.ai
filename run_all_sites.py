from pathlib import Path
from twin_model import run_pipeline

def main():
    sites = ["plant_a_high_instrumentation", "plant_b_mixed", "plant_c_legacy_heavy"]
    print(f"{'Site':<30} | {'ROC-AUC':<7} | {'OEE':<5} | {'High-Risk Alerts'}")
    print("-" * 70)
    for site in sites:
        res = run_pipeline(site_id=site)
        metrics = res["metrics"]
        biz = res["business_metrics"]
        roc = f"{metrics['roc_auc']:.3f}"
        oee = f"{biz['oee']:.3f}"
        alerts = int(biz['high_risk_alerts'])
        print(f"{site:<30} | {roc:<7} | {oee:<5} | {alerts}")

if __name__ == '__main__':
    main()
