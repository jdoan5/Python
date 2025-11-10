## Data Quality Summary
- Passed: **13**  Failed: **3**  Total: **16**

### Checks
- ❌ `min_rows` {'check': 'min_rows', 'pass': False, 'rows': 4}
- ✅ `no_duplicate_rows` {'check': 'no_duplicate_rows', 'pass': np.True_, 'duplicate_rows': 0}
- ❌ `unique_keys` {'check': 'unique_keys', 'pass': np.False_, 'duplicate_key_rows': 1, 'keys': ['customer_id']}
- ✅ `dtype` {'check': 'dtype', 'column': 'customer_id', 'target': 'int', 'pass': np.True_, 'invalid_dtype': 0}
- ✅ `not_null` {'check': 'not_null', 'column': 'customer_id', 'pass': np.True_, 'nulls': 0}
- ❌ `unique` {'check': 'unique', 'column': 'customer_id', 'pass': np.False_, 'non_unique': 2}
- ✅ `dtype` {'check': 'dtype', 'column': 'email', 'target': 'str', 'pass': True, 'invalid_dtype': 0}
- ✅ `not_null` {'check': 'not_null', 'column': 'email', 'pass': np.True_, 'nulls': 0}
- ✅ `regex` {'check': 'regex', 'column': 'email', 'pass': np.True_, 'invalid': 0}
- ✅ `dtype` {'check': 'dtype', 'column': 'signup_date', 'target': 'date', 'pass': np.True_, 'invalid_dtype': 0}
- ✅ `not_null` {'check': 'not_null', 'column': 'signup_date', 'pass': np.True_, 'nulls': 0}
- ✅ `min` {'check': 'min', 'column': 'signup_date', 'pass': np.True_, 'violations': 0, 'min': '2020-01-01'}
- ✅ `dtype` {'check': 'dtype', 'column': 'plan', 'target': 'str', 'pass': True, 'invalid_dtype': 0}
- ✅ `not_null` {'check': 'not_null', 'column': 'plan', 'pass': np.True_, 'nulls': 0}
- ✅ `allowed_values` {'check': 'allowed_values', 'column': 'plan', 'pass': np.True_, 'invalid': 0}
- ✅ `foreign_key` {'check': 'foreign_key', 'child': 'plan', 'pass': True, 'missing_count': 0}