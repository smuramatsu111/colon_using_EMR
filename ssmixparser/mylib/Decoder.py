import sys
import chardet

def byte_decode(chunks):    
    data = ''
    char_code = chardet.detect(chunks)['encoding']
    if char_code == 'utf-8':
        for codec in ['utf_8_sig', 'utf_8']:
            try: data = chunks.decode(codec)
            except: pass

    elif char_code == 'ISO-2022-JP':
        for codec in ['iso2022_jp_ext', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_3', 'iso2022_jp_2004']:
            try: data = chunks.decode(codec)
            except: pass
            
    elif char_code == 'ascii':
        try: data = chunks.decode('ascii')
        except: pass

    elif char_code == 'SHIFT_JIS':
        for codec in ['shift_jis', 'shift_jisx0213', 'shift_jis_2004']:
            try: data = chunks.decode(codec)
            except: pass
    
    if data == '': print(char_code, file=sys.stderr)

    return data
