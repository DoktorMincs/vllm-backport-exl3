# Security Policy

## Scope

`vllm-backport-exl3` is a fork of `wtdcode/vllm-backport` carrying the
MiniMax-M3.1-Flash EXL3 engine work. The engine changes live on the
`exl3-mm31-flash` branch; the EXL3 plugin changes are distributed as diffs
under `docs/exl3-mm31/patches/`. No model weights are distributed here.

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting:

<https://github.com/DoktorMincs/vllm-backport-exl3/security/advisories/new>

Please do not open a public issue for an unfixed vulnerability. Include the
affected file or commit, the configuration needed to reproduce it, and the
impact you observed.

## Upstream and third-party issues

- Vulnerabilities in upstream vLLM that this fork did not introduce belong to
  the vLLM project:
  <https://github.com/vllm-project/vllm/security/advisories/new>
- Issues in the `vcruz305/vllm-exl3` plugin itself belong to that project:
  <https://github.com/vcruz305/vllm-exl3>
- Issues in the engine changes, the documentation, or the scripts in this
  repository are in scope here.

## Disclosure

We will acknowledge a report as soon as possible and coordinate a fix and
disclosure with the reporter. Please allow time for a fix before publishing
details.
