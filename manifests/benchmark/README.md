# Frozen benchmark suites

These suites are copied unchanged from the original closed-loop benchmark.
The hierarchy is `<suite>/<tier>/plan.yaml` with relative scenario files.
The default evaluator discovers these plans. `--max_episodes` limits each suite/tier;
it does not add episodes or rebalance a prefix. Use a fresh plan for an exact budget.
