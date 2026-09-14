# ADR-004: Enterprise Data Governance, Telemetry Privacy & Air-Gapped Operation

## Context
Deep learning models are increasingly trained on confidential, proprietary, or regulated data (HIPAA health records, GDPR personally identifiable information, internal financial datasets). Using external optimizer frameworks can introduce compliance concerns if the library:
1. Emits outbound network calls (analytics, crash reporting, license checks).
2. Buffers unencrypted training samples to local temporary files.
3. Exposes unauthenticated web endpoints for monitoring.

## Decision
AcceleratorAI enforces an architectural guarantee of **Zero-Data Exfiltration**:
1. **Local & Ephemeral**: Tensors exist solely in volatile GPU/CPU RAM. No file caching of batches occurs.
2. **Air-Gapped Operation**: Zero network socket calls are made by the core library. All external network access is strictly prohibited.
3. **Telemetry Sanitization**:
   - `EngineTelemetry` exports only aggregate scalar indicators (RPM, boost pressure, learning rate, rounded loss).
   - No individual sample values, embeddings, or gradient vectors are serialized.
   - Loss values are rounded to 4 decimal places to prevent loss-trajectory membership inference attacks.
4. **Hardened Cockpit Server**:
   - Default binding to `127.0.0.1` (localhost only).
   - Token-based authentication (`X-Cockpit-Token` / Bearer).
   - Strict CORS origin validation.
   - Boundary checks on all remote actions (`0 <= boost_psi <= 45`).

## Consequences
- **Positive**: Enterprise organizations and security teams can audit and deploy AcceleratorAI with zero compliance friction.
- **Positive**: Fully compliant with GDPR Article 5 (Data Minimization) and confidential computing standards.
