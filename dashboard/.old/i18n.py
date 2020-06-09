import yaml
from pathlib import Path


def translate(lang):
    lang = {'🇬🇧 English': 'en', '🇪🇸 Español': 'es'}.get(lang, lang)    

    def msg(text_en, text_es):
        if lang == 'en':
            return text_en

        return text_es

    return msg
