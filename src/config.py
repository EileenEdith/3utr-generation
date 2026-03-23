"""Compatibility config shim for GEMORNA shared-library/checkpoint loading.

This file is intentionally top-level because the upstream `libg2m.so` binary
expects to import `config` from the original GEMORNA layout. To keep the local
project structure simple, we keep the minimal compatibility definitions here and
use this file as the single runtime config source.
"""

from dataclasses import dataclass

MEAN = 0.0
STD = 0.02
init_token = '<sos>'
eos_token = '<eos>'


@dataclass
class GEMORNA_CDS_Config:
    input_dim: int = 29
    output_dim: int = 347
    hidden_dim: int = 128
    num_layers: int = 12
    num_heads: int = 8
    ff_dim: int = 256
    dropout: int = 0.1
    cnn_kernel_size: int = 3
    cnn_padding: int = 1
    prot_pad_idx: int = 1
    cds_pad_idx: int = 1


@dataclass
class GEMORNA_5UTR_Config:
    block_size: int = 768
    vocab_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 144
    dropout: float = 0.1
    bias: bool = True


@dataclass
class GEMORNA_3UTR_Config:
    block_size: int = 1024
    vocab_size: int = 448
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 288
    dropout: float = 0.1
    bias: bool = True


five_prime_utr_vocab = {}

three_prime_utr_vocab = {'<sos>': 1, 'AGC': 2, 'AAG': 3, 'GGC': 4, 'AGA': 5, 'AUG': 6, 'AAA': 7, 'GCA': 8, 'CUG': 9, 'UGC': 10, 'GCU': 11, 'UCC': 12, 'CAU': 13, 'UAA': 14, 'UUC': 15, 'CUU': 16, 'CCC': 17, 'UGU': 18, 'GUU': 19, 'GGU': 20, 'GGG': 21, 'GGA': 22, 'GAC': 23, 'CAA': 24, 'CGC': 25, 'GAA': 26, 'CAC': 27, 'GUA': 28, 'CCU': 29, 'GCC': 30, 'UGG': 31, 'AUU': 32, 'UCG': 33, 'GAU': 34, 'ACA': 35, 'UAC': 36, 'CUC': 37, 'AAC': 38, 'UGA': 39, 'ACU': 40, 'CAG': 41, 'AGU': 42, 'AAU': 43, 'ACG': 44, 'AGG': 45, 'UUG': 46, 'CAN': 47, '<eos>': 48, 'GAG': 49, 'CGA': 50, 'UCU': 51, 'ACC': 52, 'AUC': 53, 'GUC': 54, 'GUG': 55, 'CCA': 56, 'UAU': 57, 'CGG': 58, 'UUU': 59, 'UUA': 60, 'AUA': 61, 'CUA': 62, 'GCG': 63, 'UCA': 64, 'CGU': 65, 'UNN': 66, 'UAG': 67, 'CCG': 68, 'UGN': 69, 'GUN': 70, 'CUN': 71, 'UUN': 72, 'ACN': 73, 'GCN': 74, 'ANN': 75, 'AUN': 76, 'AGN': 77, 'CCN': 78, 'GNN': 79, 'AAN': 80, 'UAN': 81, 'CNN': 82, 'UCN': 83, 'GAN': 84, 'UNU': 85, 'GGN': 86, 'CGN': 87, 'NNN': 88, 'NNA': 89, 'NGA': 90, 'NGG': 91, 'NCC': 92, 'GNG': 93, 'ANC': 94, 'ANU': 95, 'NAU': 96, 'NUU': 97, 'NNC': 98, 'NCA': 99, 'NAA': 100, 'NNU': 101, 'NAC': 102, 'NUA': 103, 'NCG': 104, 'NCU': 105, 'NUC': 106, 'NGC': 107, 'UNC': 108, 'NNG': 109, 'NUG': 110, 'ANG': 111, 'NAG': 112, 'NGU': 113, 'CNU': 114, 'NAN': 115, 'ANA': 116, 'CNC': 117, 'UNG': 118, 'UNA': 119, 'CNA': 120, 'GNU': 121, 'NUN': 122, 'GNC': 123, 'CNG': 124, 'GNA': 125, 'NGN': 126, 'NCN': 127, 'ARU': 128, 'GUW': 129, 'CUK': 130, 'GGR': 131, 'UMU': 132, 'YUU': 133, 'AWA': 134, 'GUY': 135, 'YCC': 136, 'CGR': 137, 'MCG': 138, 'RUC': 139, 'RGA': 140, 'CCK': 141, 'GCM': 142, 'YUC': 143, 'KGC': 144, 'CRG': 145, 'GCY': 146, 'AYA': 147, 'ARA': 148, 'UUR': 149, 'GRC': 150, 'CWC': 151, 'YGA': 152, 'RGG': 153, 'CKG': 154, 'GGS': 155, 'KGU': 156, 'AMA': 157, 'CYG': 158, 'CCY': 159, 'CCR': 160, 'GAR': 161, 'GSC': 162, 'GCW': 163, 'AKC': 164, 'AUR': 165, 'UAW': 166, 'UGY': 167, 'SUC': 168, 'KAA': 169, 'URA': 170, 'ARC': 171, 'GKA': 172, 'CRA': 173, 'GYU': 174, 'UYG': 175, 'UUY': 176, 'AYG': 177, 'AYU': 178, 'YAG': 179, 'URC': 180, 'CWG': 181, 'MUC': 182, 'GWC': 183, 'USU': 184, 'RUU': 185, 'CAY': 186, 'AKG': 187, 'RCU': 188, 'CRU': 189, 'UUW': 190, 'YAA': 191, 'RGU': 192, 'GAS': 193, 'URU': 194, 'CKC': 195, 'RUG': 196, 'YUA': 197, 'UGR': 198, 'AAW': 199, 'WCC': 200, 'AUY': 201, 'AWC': 202, 'UWC': 203, 'GCS': 204, 'RAC': 205, 'RAG': 206, 'CMC': 207, 'GYG': 208, 'RUA': 209, 'YCU': 210, 'AUW': 211, 'CWU': 212, 'YAC': 213, 'UCY': 214, 'GSG': 215, 'SAU': 216, 'WUU': 217, 'RCG': 218, 'RCC': 219, 'GAY': 220, 'CUY': 221, 'CAR': 222, 'RAU': 223, 'KAC': 224, 'MGA': 225, 'UYA': 226, 'YGC': 227, 'CCW': 228, 'ACY': 229, 'CMG': 230, 'KCU': 231, 'YUG': 232, 'CUS': 233, 'GUR': 234, 'KNN': 235, 'KGG': 236, 'CSG': 237, 'CYC': 238, 'YGU': 239, 'ACK': 240, 'AAR': 241, 'KCA': 242, 'ACS': 243, 'RAA': 244, 'MGU': 245, 'AGK': 246, 'MGC': 247, 'UWA': 248, 'SUU': 249, 'KCG': 250, 'GRA': 251, 'MGG': 252, 'CGK': 253, 'YCA': 254, 'AAM': 255, 'SCC': 256, 'CUR': 257, 'CYU': 258, 'WGC': 259, 'UYU': 260, 'UGS': 261, 'CSA': 262, 'GCR': 263, 'UGK': 264, 'GMC': 265, 'UWW': 266, 'YGG': 267, 'RCA': 268, 'MAU': 269, 'GRU': 270, 'CYA': 271, 'GGY': 272, 'UYC': 273, 'SUG': 274, 'SAG': 275, 'UAY': 276, 'ACR': 277, 'KCC': 278, 'UKC': 279, 'URG': 280, 'UCR': 281, 'GYA': 282, 'AAK': 283, 'GYR': 284, 'GSU': 285, 'ASA': 286, 'GYC': 287, 'WGG': 288, 'CGY': 289, 'UCW': 290, 'YAU': 291, 'WUG': 292, 'AWG': 293, 'UUM': 294, 'CSC': 295, 'SMA': 296, 'ARG': 297, 'GCK': 298, 'YCG': 299, 'AGM': 300, 'GRG': 301, 'ACM': 302, 'UKU': 303, 'KUG': 304, 'KUK': 305, 'RGC': 306, 'WAU': 307, 'GKG': 308, 'UMC': 309, 'GKU': 310, 'AKA': 311, 'AGY': 312, 'CMU': 313, 'AYC': 314, 'MUG': 315, 'AUM': 316, 'CSU': 317, 'AUK': 318, 'UAR': 319, 'UGM': 320, 'KUC': 321, 'WCA': 322, 'GUM': 323, 'AMC': 324, 'SAA': 325, 'AGR': 326, 'CRC': 327, 'GWU': 328, 'WAG': 329, 'CAW': 330, 'CUW': 331, 'AAY': 332, 'UWU': 333, 'MUA': 334, 'ASC': 335, 'ASG': 336, 'CAM': 337, 'KAG': 338, 'AAS': 339, 'HGG': 340, 'UKA': 341, 'UAS': 342, 'CCM': 343, 'MAA': 344, 'GWA': 345, 'WUC': 346, 'MWY': 347, 'URW': 348, 'KAU': 349, 'AKU': 350, 'AMG': 351, 'UWG': 352, 'GWG': 353, 'KUA': 354, 'CGW': 355, 'CKU': 356, 'UUK': 357, 'GKC': 358, 'AMU': 359, 'UCK': 360, 'USG': 361, 'GMG': 362, 'CKA': 363, 'MCU': 364, 'KKU': 365, 'AGS': 366, 'UMG': 367, 'CAK': 368, 'AGW': 369, 'UMA': 370, 'MRU': 371, 'KGA': 372, 'GMU': 373, 'WUA': 374, 'GGK': 375, 'UAK': 376, 'KUU': 377, 'GUS': 378, 'WAA': 379, 'UAM': 380, 'UCS': 381, 'WAC': 382, 'SGA': 383, 'WCU': 384, 'CWA': 385, 'SGU': 386, 'CAS': 387, 'MCC': 388, 'AUS': 389, 'SCU': 390, 'CUM': 391, 'GAK': 392, 'SAC': 393, 'MUU': 394, 'CCS': 395, 'SUA': 396, 'ASU': 397, 'GMA': 398, 'WGU': 399, 'GGM': 400, 'UKG': 401, 'AWU': 402, 'MAG': 403, 'UCM': 404, 'MCA': 405, 'UUS': 406, 'CMA': 407, 'GUK': 408, 'GAM': 409, 'GAW': 410, 'YKA': 411, 'ARK': 412, 'CGS': 413, 'UGW': 414, 'USA': 415, 'UKR': 416, 'VAG': 417, 'AAD': 418, 'GKY': 419, 'WGA': 420, 'SGC': 421, 'CGM': 422, 'GGW': 423, 'UYN': 424, 'WWA': 425, 'WCG': 426, 'SYA': 427, 'MAC': 428, 'SCA': 429, 'RUN': 430, 'SNN': 431}
