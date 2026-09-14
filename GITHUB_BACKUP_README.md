# GitHub research backup

Private repository snapshot, 2026-09-14. Code, papers, reports and small outputs are tracked in Git. Large outputs, datasets, generated representations and checkpoints are in the release research-backup-2026-09-14.

Download every research-data-*.zip from that release and extract all ZIPs into the repository root. Each ZIP is independently extractable and preserves original relative paths. GITHUB_BACKUP_MANIFEST.json lists the original size and SHA-256 of every included file and its storage location. The release checksum file verifies ZIPs.

Excluded only local Python virtual environments, bytecode/test caches and nested third-party .git histories. Dependencies can be recreated from requirements files. Dataset and vendored-code licenses remain those of their sources. No paper or experimental result was modified for this backup.

Release asset limits: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

After extracting the release assets, run: python GITHUB_BACKUP_VERIFY.py

