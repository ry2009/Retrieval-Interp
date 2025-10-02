# Shadeform Provisioning Guide

## 1. List Affordable GPUs
```bash
export SHADEFORM_API_KEY="<api_key>"
curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" \
  'https://api.shadeform.ai/v1/instances/types?sort=price&available=true' \
  | jq '[.instance_types[] | select(.hourly_price <= 60) | {type: .shade_instance_type, price: .hourly_price, regions: [.availability[] | select(.available==true) | .region]}]'
```

## 2. Launch A30 Prototype Node
```bash
export SHADEFORM_API_KEY="<api_key>"
curl -s -X POST https://api.shadeform.ai/v1/instances/create \
  -H "X-API-KEY: $SHADEFORM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "cloud": "massedcompute",
    "region": "wichita-usa-1",
    "shade_instance_type": "A30",
    "shade_cloud": true,
    "name": "llm-search-a30-<date>",
    "os": "ubuntu22.04_cuda12.6_shade_os",
    "ssh_key_id": "<ssh_key_id>"
  }'
```
Record the `id` from the response for status checks.

## 3. Poll Status & Retrieve Connection Details
```bash
curl -s -H "X-API-KEY: $SHADEFORM_API_KEY" https://api.shadeform.ai/v1/instances \
  | jq '.instances[] | select(.id == "<instance_id>")'
```
Wait for `status` to become `active` and capture `ip`, `ssh_user`, and `ssh_port`.

## 4. Connect via SSH
Download the private key associated with `<ssh_key_id>` from the Shadeform console, restrict permissions, then:
```bash
chmod 600 shadeform_instance.pem
ssh -i shadeform_instance.pem shadeform@<ip>
```

## 5. Shutdown When Idle
```bash
curl -s -X POST https://api.shadeform.ai/v1/instances/<instance_id>/delete \
  -H "X-API-KEY: $SHADEFORM_API_KEY"
```

## Notes
- An A30 at $33/hour offers ~18 minutes of runtime within a $10 balance.
- Always verify wallet balance (`Billing` tab) prior to longer runs.
- Consider scripting `auto_stop` timers once workloads stabilize.
