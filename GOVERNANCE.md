# Governance

GPU Systems Lab is maintained by [@edcadet10](https://github.com/edcadet10). The
maintainer sets release scope, reviews changes, manages repository settings, and is
accountable for applying the published benchmark and conduct policies consistently.

## Decision process

- Small fixes and documentation changes are decided in pull-request review.
- New public APIs, dependencies, benchmark protocols, or kernel families begin with an
  issue or discussion and remain open for community input for at least seven days when
  no security or correctness urgency exists.
- Decisions prioritize reproducibility, explicit contracts, maintainability, and
  evidence over headline benchmark numbers.
- Material objections and the resolution are summarized in the issue or pull request.

## Releases

Releases use semantic versioning for the Python API. Benchmark data is additionally
tied to a commit, environment, shape, and protocol; a package version does not make
measurements portable across hardware.

## Protected branch operations

Changes normally reach `main` through a pull request after all required checks pass.
External contributions also require maintainer review. Because the repository has one
maintainer and a pull-request author cannot approve their own change, administrator
bypass remains available for maintainer-authored release and recovery work. It is used
only after required checks pass, and the bypass is identified in the pull request or
release record so the exception is reviewable.

## Becoming a maintainer

Sustained contributors may be invited to maintain an area after multiple reviewed
contributions and demonstrated care with numerical correctness, measurement claims,
security, and community review. Repository administration remains least-privilege and
can be revoked if access is unused or the conduct policy is violated.
