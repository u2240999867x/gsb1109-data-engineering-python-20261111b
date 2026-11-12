"""Export a Voluptuous inbound contract to JSON Schema.

Run with: python examples/json_schema_export.py
"""

import json

from voluptuous import (
    All,
    Any,
    Coerce,
    Exclusive,
    Match,
    Optional,
    Required,
    Schema,
)
from voluptuous.json_schema import to_json_schema

CONTRACT = Schema(
    {
        Required('order_id'): str,
        Required('priority'): int,
        Optional('state'): Any('new', 'queued', 'done'),
        Optional('tags'): [str],
        Optional('channel'): All(str, Match(r'^(web|app|pos)-')),
        Optional('retries'): Coerce(int),
        Exclusive('coupon', 'discount'): str,
        Exclusive('gift_card', 'discount'): str,
        Match(r'^meta_'): str,
    }
)

if __name__ == '__main__':
    result = to_json_schema(CONTRACT)
    print('=== JSON Schema ===')
    print(json.dumps(result.schema, indent=2, sort_keys=True))
    print()
    print('=== Coverage report ===')
    print(result.summary())
