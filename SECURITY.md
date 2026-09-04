# Security policy

## Supported versions

Security fixes are applied to the current `1.x` line.

## Reporting

Report a suspected vulnerability privately to the repository maintainers rather than opening a public issue. Include the affected version, minimal reproduction, expected impact, and any proposed mitigation.

## Cryptographic scope

The keyed whitening, pseudo-noise generation, CRC, and truncated HMAC in this repository provide deterministic carrier secrecy and payload authentication. They do **not** encrypt the host media, provide public-key ownership proofs, implement DRM, or replace a reviewed cryptographic protocol. Protect watermark keys as production secrets and rotate them if exposure is suspected.
