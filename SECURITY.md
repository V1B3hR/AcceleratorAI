# 🛡️ Security Policy

## Supported Versions

We actively provide security updates and patches for the following versions of **AcceleratorAI**:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5.0 | :x:                |

---

## Reporting a Vulnerability

The AcceleratorAI team takes security seriously. If you discover a vulnerability, please disclose it responsibly so we can remediate it before public disclosure.

### How to Report:
1. **Confidential Reporting**: Please email security concerns to `security@acceleratorai.org` (or contact the maintainers via GitHub Security Advisories at https://github.com/V1B3hR/AcceleratorAI/security/advisories/new).
2. **Include in Report**:
   * Description of the vulnerability and attack vector (e.g., DoS, memory leak, CSRF, input validation bypass).
   * Minimal reproducible script or proof-of-concept (PoC).
   * Affected components (`InputGuard`, `live_cockpit_server.py`, `DistributedECUCoordinator`, etc.).
   * Hardware & software environment (Python version, PyTorch version, CUDA version).

### What to Expect:
* **Initial Response**: Within 48 hours confirming receipt of the advisory.
* **Assessment & Patch**: We will provide an estimated timeline for a patch or mitigation.
* **Credit**: We will gladly acknowledge your responsible disclosure in our release notes (unless you prefer anonymity).

---

## Security Best Practices for Deployments

1. **Keep Cockpit Local**: Do not expose `live_cockpit_server.py` to untrusted public networks without reverse proxy authentication (e.g. Nginx with TLS and OAuth2/JWT).
2. **Enable InputGuard Strict Mode**: When training on uncurated or user-submitted data, instantiate `InputGuard(strict_mode=True)` to reject malformed or extreme tensors.
3. **Use Distributed Timeouts**: In multi-GPU clusters, ensure `DistributedECUCoordinator(timeout_seconds=10.0)` is configured to prevent collective lockups.
