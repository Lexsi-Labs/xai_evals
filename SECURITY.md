# Security Policy

## Supported Versions

Security fixes are released for the latest published version of xai_evals only. Please upgrade to the latest release before reporting an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately using GitHub's private vulnerability reporting: open the [advisory form](https://github.com/Lexsi-Labs/xai_evals/security/advisories/new), or go to the repository's **Security** tab and choose **Report a vulnerability**.

### What to Include

1. A description of the vulnerability
2. Steps to reproduce the issue
3. The potential impact
4. Affected versions (if known)
5. A suggested fix (if you have one)

### What to Expect

- Acknowledgment of your report within 7 days
- An assessment of severity and an update on our progress
- A fix released as a patch version
- Credit for the report in the advisory, with your permission

## Disclosure Policy

We will prepare a security advisory, release a patch, and disclose the issue publicly after users have had time to update.

## Security Practices

- Never commit API keys or tokens.
- Only load models and datasets from sources you trust; pickled models can run arbitrary code when loaded.
- Dependencies and GitHub Actions are monitored with Dependabot, and the code is scanned with CodeQL.
