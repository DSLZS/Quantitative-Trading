"""V165 Runner Script"""
import argparse
from src.alpha_research_v165 import V165Runner

def main():
    parser = argparse.ArgumentParser(description='Run V165 Audit')
    parser.add_argument('--year', type=int, default=2024, help='Year to audit')
    args = parser.parse_args()
    
    runner = V165Runner(output_dir='reports')
    result = runner.run_audit(args.year)
    
    print("\n" + "=" * 70)
    print(f"V165 Audit Summary - Year {args.year}")
    print("=" * 70)
    
    t1_ic = result.get('t1_ic', {})
    ic_decay = result.get('ic_decay', {})
    
    print(f"T+1 Rank IC: {t1_ic.get('mean_ic', 0):.4f} (Target: > 0.095)")
    print(f"IC IR: {t1_ic.get('ic_ir', 0):.2f} (Target: > 0.60)")
    print(f"IC Decay: T+1({ic_decay.get('t1_ic', 0):.4f}) -> T+3({ic_decay.get('t3_ic', 0):.4f}) -> T+5({ic_decay.get('t5_ic', 0):.4f})")
    print(f"Monotonic: {ic_decay.get('is_monotonic', False)}")
    print(f"Overall: {'PASSED ✓' if result.get('passed', False) else 'FAILED ✗'}")
    print("=" * 70)

if __name__ == '__main__':
    main()