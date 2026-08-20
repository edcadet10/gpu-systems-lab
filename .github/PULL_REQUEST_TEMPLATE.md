## Contract

What numerical, API, or documentation contract changes? What remains unchanged?

## Claim and kill criterion

State the claim this change depends on and the one pre-registered observation that
would make it false. Write `not applicable` for a change with no empirical claim.

## Evidence and test

Link the issue/discussion and list the exact checks run. For performance work, attach
all raw result files and profiler evidence required by `docs/benchmarking.md`.

## Boundaries

List unsupported shapes, dtypes, devices, layouts, gradient behavior, or generalization
limits introduced or discovered by this change.

## Checklist

- [ ] I added or updated tests for the contract and negative cases.
- [ ] I updated the relevant documentation.
- [ ] I ran `ruff check .`, `ruff format --check .`, and `pytest`.
- [ ] I did not add secrets, proprietary artifacts, or unauthorized benchmark data.
- [ ] My contribution can be distributed under Apache-2.0.
