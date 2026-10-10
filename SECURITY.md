# Security Policy

PITWALL AI is a small, volunteer-maintained educational project. There is no dedicated security team and no
guaranteed response time, but reports are taken seriously and will be handled on a best-effort basis.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Instead, use GitHub's private
"Report a vulnerability" feature on the repository's *Security* tab (if enabled), or contact the repository owner
through the contact details on their GitHub profile. Include what you found, steps to reproduce, and the
affected version.

## Scope notes

* The app runs locally and does not handle user accounts or secrets. No API keys are required or stored.
* Model files are loaded with `joblib` (pickle-based). Only load model files you created yourself or trust.
