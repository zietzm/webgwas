```bash
aws iam create-user --user-name hetzner-sync-user
aws iam attach-user-policy --user-name hetzner-sync-user --policy-arn arn:aws:iam::069134179952:policy/WebGWAS-policy
aws iam create-access-key --user-name hetzner-sync-user
```

This was saved into `aws-hetzner-sync-user-key.json`.

I manually wrote this to `~/.aws/credentials` on the Hetzner machine in TOML format.

```toml
[default]
aws_access_key_id = ACCESS_KEY
aws_secret_access_key = SECRET_KEY
```
