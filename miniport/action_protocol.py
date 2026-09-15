"""Single-action response boundaries and actionable parsing errors."""
import json


def complete_object(text):
    text = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


class StopAfterAction:
    """Stop generation as soon as a complete top-level JSON object is emitted."""
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # The pilot generates one response at a time. Decode generated tokens only;
        # braces and escaped quotes inside Python source are handled by JSON itself.
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:].tolist(), skip_special_tokens=True)
        return complete_object(text)


def parse_action(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg == 'Extra data':
            stripped = text.lstrip()
            first, end = json.JSONDecoder().raw_decode(stripped)
            suffix = stripped[end:].strip()
            try:
                second, _ = json.JSONDecoder().raw_decode(suffix)
            except json.JSONDecodeError:
                second = None
            if isinstance(first, dict) and isinstance(second, dict):
                raise ValueError('You emitted multiple actions. Nothing was applied. Return only one JSON action. '
                                 'Edits are tested automatically; do not append a test or stop action.') from exc
            raise ValueError('Extra content follows the JSON action. Nothing was applied. '
                             'Return one JSON action and end the response; edits are tested automatically.') from exc
        raise
