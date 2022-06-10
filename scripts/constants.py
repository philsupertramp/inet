"""
Commonly used constant values

.. data:: COLEOPTERA_IDS

"""

## ID range of Coleopterans
COLEOPTERA_IDS = [i for i in range(184, 418 + 1)]
## ID range of Hemipterans
HEMIPTERA_IDS = [i for i in range(507, 670 + 1)]
## ID range of Formicidaes
HYMENOPTERA_IDS = [i for i in range(738, 771 + 1)]
## ID range of Lepidopterans
LEPIDOPTERA_IDS = [i for i in range(848, 2277 + 1)]
## ID range of Ordinatas
ORDONATA_IDS = [i for i in range(2304, 2594 + 1)]

## Enum for Coleopterans
COLEOPTERA = 0
## Enum for Hemipterans
HEMIPTERA = 2
## Enum for Formicidaes
HYMENOPTERA = 1
## Enum for Lepidopterans
LEPIDOPTERA = 4
## Enum for Ordonatas
ORDANATA = 3


## String repr. for Coleopterans
COLEOPTERA_STR = 'Coleoptera'
## String repr. for Coleopterans
HEMIPTERA_STR = 'Hemiptera'
## String repr. for Hymenopterans
HYMENOPTERA_STR = 'Hymenoptera>Formicidae'
## String repr. for Lepidopterans
LEPIDOPTERA_STR = 'Lepidoptera'
## String repr. for Ordonatas
ORDONATA_STR = 'Odonata'


## Class ID range map. name: ID-range
CLASS_RANGES = {
    COLEOPTERA_STR: COLEOPTERA_IDS,
    HEMIPTERA_STR: HEMIPTERA_IDS,
    HYMENOPTERA_STR: HYMENOPTERA_IDS,
    LEPIDOPTERA_STR: LEPIDOPTERA_IDS,
    ORDONATA_STR: ORDONATA_IDS,
}

## Class internal-ID map. name: internal-ID
CLASS_MAP = {
    COLEOPTERA_STR: COLEOPTERA,
    HYMENOPTERA_STR: HYMENOPTERA,
    HEMIPTERA_STR: HEMIPTERA,
    ORDONATA_STR: ORDANATA,
    LEPIDOPTERA_STR: LEPIDOPTERA,
}

## Class name map. Internal-ID: name
LABEL_MAP = {
    COLEOPTERA: COLEOPTERA_STR,
    HEMIPTERA: HEMIPTERA_STR,
    HYMENOPTERA: HYMENOPTERA_STR,
    LEPIDOPTERA: LEPIDOPTERA_STR,
    ORDANATA: ORDONATA_STR,
}
