SECS = 'Secs'

TIME = 'Time'

DP = 'Dp'

LDP = 'lDp'

DNDLDP = 'norm_conc'

FLAGS = 'Flags'

MESSAGE = 'Message'

P_ION = 'p_ion'
N_ION = 'n_ion'
P_PAR = 'p_par'
N_PAR = 'n_par'

MODE = 'mode'

POL_DIC = {
    P_ION: 'p',
    N_ION: 'n',
    P_PAR: 'p',
    N_PAR: 'n',
}

MODE_DIC = {
    P_ION: 'nds',
    N_ION: 'nds',
    P_PAR: 'np',
    N_PAR: 'np',
}

MODES = [
    P_ION,
    N_ION,
    P_PAR,
    N_PAR,
]

FLAG_ID = 'flag_id'

_mode_desc = '''
P_ION=positive ions
N_ION=negative ions
P_PAR=positive particles
P_NEG=negatice particles
'''
META = {
    # P_ION: {
    #     'long_name': 'positive ions',
    #     'units': '',
    #     'description': ''
    # },
    # N_ION: {
    #     'long_name': 'negative ions',
    #     'units': '',
    #     'description': ''
    # },
    # P_PAR: {
    #     'long_name': 'positive particles',
    #     'units': '',
    #     'description': ''
    # },
    # N_PAR: {
    #     'long_name': 'negatice particles',
    #     'units': '',
    #     'description': ''
    # },
    MODE: {
        'long_name': 'nais mode',
        'units': '',
        'description': _mode_desc
    },
    TIME: {
        'long_name': 'UTC time',
        # 'units': '',
        'description': 'center aligned'
    },
    DP: {
        'long_name': 'particle diameter',
        'units': 'm',
        'description': 'geometric center of the bin'
    },
    DNDLDP: {
        'long_name': 'normalized concentration',
        'units': 'dN/dlogDp cm-3',
        'description': ''
    },
    FLAG_ID: {
        'long_name': 'flag id',
        'units': '',
        'description': 'two character unit identifier'
    },
    MESSAGE : {
        'long_name': 'flag id -> message ',
        'units': '',
        'description': 'dictionary between flag id and message'
    },

}
