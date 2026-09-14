---
name: Security report
about: Report a potential vulnerability or security concern in AcceleratorAI
title: '[SEC] '
labels: 'security'
assignees: ''
---

> [!CAUTION]
> If this is a critical remote code execution, telemetry leakage, or privilege escalation vulnerability, please report it confidentially via email or GitHub Security Advisories as described in [SECURITY.md](https://github.com/V1B3hR/AcceleratorAI/blob/main/SECURITY.md).

**Component Affected**
- [ ] InputGuard / Tensor Sanitizer
- [ ] Live Cockpit Web Server (`live_cockpit_server.py`)
- [ ] Distributed ECU Coordinator / NCCL Communication
- [ ] Checkpoint Deserialization / `state_dict`
- [ ] Dependencies / Supply Chain

**Vulnerability Description**
A clear description of the potential vulnerability.

**Impact & Threat Model**
Who is affected? What could an attacker achieve (e.g. DoS, training divergence, telemetry exfiltration)?

**Proof of Concept / Reproducible Steps**
```python
# Minimal demonstration snippet
```

**Proposed Mitigation / Fix**
Any ideas on how to remediate the vulnerability.
