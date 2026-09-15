"""Policy-visible runtime errors and separately identified output validation."""
import re

RUNTIME_CODES = {'runtime_error', 'missing_attribute', 'missing_key', 'type_error',
                 'not_implemented', 'output_type_mismatch'}
STAGES = {'import', 'build', 'topology', 'load_parameters', 'export_parameters',
          'run', 'layer', 'output', 'repeat', 'run_cached'}


def runtime_diagnostic(reply):
    raw = reply.get('diagnostic')
    if not isinstance(raw, dict):
        return None
    code = raw.get('code')
    if code not in RUNTIME_CODES:
        code = 'runtime_error'
    result = {'code': code, 'origin': 'candidate_runtime'}
    if isinstance(raw.get('message'), str):
        result['message'] = raw['message']
    if code == 'output_type_mismatch':
        result['origin'] = 'output_validation'
        if isinstance(raw.get('field'), str):
            result['field'] = raw['field']
    if raw.get('stage') in STAGES:
        result['stage'] = raw['stage']
    if type(raw.get('line')) is int and 0 < raw['line'] <= 128000:
        result['line'] = raw['line']
    for field in ('exception_type', 'attribute', 'object_type', 'key', 'argument', 'required_type', 'actual_type'):
        value = raw.get(field)
        if isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,63}', value):
            result[field] = value
    return result


class VerificationIssue(ValueError):
    def __init__(self, code, **details):
        self.diagnostic = {'code': code, **details}
        super().__init__(code)
