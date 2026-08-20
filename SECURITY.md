# Security policy

## Supported versions

The `main` branch and latest tagged release receive security fixes. Earlier snapshots
may be patched when practical but are not guaranteed support.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use the repository's
**Security → Report a vulnerability** flow to create a private security advisory:

https://github.com/edcadet10/gpu-systems-lab/security/advisories/new

Include affected versions, impact, reproduction steps, and any suggested mitigation.
You should receive an acknowledgment within five business days. The maintainer will
coordinate validation, a fix, disclosure timing, and credit with the reporter.

## GPU runner policy

This public repository does not execute pull-request code on a persistent self-hosted
GPU runner. GPU measurements must run in isolated, disposable infrastructure without
repository or cloud secrets. Submitted result data is reviewed separately from code.

Never attach credentials, cloud metadata, proprietary model artifacts, or unredacted
host inventories to an issue, discussion, benchmark JSON file, or profiler report.
