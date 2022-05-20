"""
Dataset split into 4 classes with id ranges:
- Coleoptera: [184, 418]
- Hypenoptera: [738, 771]
- Lepidoptera: [848, 2277]
- Ordonata: [2304, 2594]
- Hemiptera [507, 670]
"""

COLEOPTERA_IDS = [i for i in range(184, 418 + 1)]
HEMIPTERA_IDS = [i for i in range(507, 670 + 1)]
HYMENOPTERA_IDS = [i for i in range(738, 771 + 1)]
LEPIDOPTERA_IDS = [i for i in range(848, 2277 + 1)]
ORDONATA_IDS = [i for i in range(2304, 2594 + 1)]

COLEOPTERA = 0
HEMIPTERA = 2
HYMENOPTERA = 1
LEPIDOPTERA = 4
ORDANATA = 3

COLEOPTERA_STR = 'Coleoptera'
HEMIPTERA_STR = 'Hemiptera'  # >Heteroptera'
HYMENOPTERA_STR = 'Hymenoptera>Formicidae'
LEPIDOPTERA_STR = 'Lepidoptera'
ORDONATA_STR = 'Odonata'

CLASS_RANGES = {
    COLEOPTERA_STR: COLEOPTERA_IDS,
    HEMIPTERA_STR: HEMIPTERA_IDS,
    HYMENOPTERA_STR: HYMENOPTERA_IDS,
    LEPIDOPTERA_STR: LEPIDOPTERA_IDS,
    ORDONATA_STR: ORDONATA_IDS,
}

CLASS_MAP = {
    COLEOPTERA_STR: COLEOPTERA,
    HYMENOPTERA_STR: HYMENOPTERA,
    HEMIPTERA_STR: HEMIPTERA,
    ORDONATA_STR: ORDANATA,
    LEPIDOPTERA_STR: LEPIDOPTERA,
}

LABEL_MAP = {
    COLEOPTERA: COLEOPTERA_STR,
    HEMIPTERA: HEMIPTERA_STR,
    HYMENOPTERA: HYMENOPTERA_STR,
    LEPIDOPTERA: LEPIDOPTERA_STR,
    ORDANATA: ORDONATA_STR,
}
