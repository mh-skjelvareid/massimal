# Massimal annotation definitions

## Classes
The following hierarchy of classes is used in annotations in Massimal. Classes are structured from a few "base" classes into more detailed classes. During annotation, the most detailed class possible is used. 

Some parent classes currently have only one child class. In this case, only one child class of interest has been defined. The structure enables using a detailed class when possible, and a more general class when needed, and also allows for later expansion.

1. Deep water
2. Algae
    1. Brown algae
        1. Kelp
            1. Laminaria hyperborea
            2. Laminaria digitata
            3. Sacchoriza polyides
            4. Saccharina latissima
            5. Alaria esculenta
        2. Rockweed
            1. Rockweed, hydrolittoral
                1. Ascophyllum nodosum
                2. Fucus vesiculosus 
                3. Fucus serratus
                4. Halidrys siliquosa
            2. Rockweed, geolittoral 
                1. Fucus spiralis
                2. Pelvetia canaliculata
        3. Other brown algae
            1. Chorda filum 
            2. Desmarestia aculeata
    2. Green algae
    3. Red algae
        1. Coralline algae 
            1. Maerl
    5. Turf
3. Seagrass
    1. Zostera marina
4. Substrate
    1. Rock
        1. Bedrock
        2. Boulder
        3. Cobble
        4. Gravel
    2. Sediment
        1. Sand
        2. Mud
5. Animals
    1. Mussels
        1. Mytilus edilus
6. Human activity 
    1. Trawl tracks



## Attributes
- **Density**: 
    - Density expressed as percentage value of seabed coverage. 100% = complete coverage, 50% = half of pixels show seabed.
    - Most relevant for seagrass 
- **Presence of turf**: 
    - Presence of turf algae (filamentous algae) growing on other vegetation. Divided into "light" (some turf algae visible) and "heavy" (visual impression dominated by turf)
    - Indicator of poor condition 
- **Presence of small animals**: 
    - Presence of small animals on vegetation (e.g. mussels, brittle stars, etc.). Expressed as "light" (some animals visible) and "heavy" (visual impression dominated by the animals).
    - Indicator of condition(?)
- **Dead**: 
    - Indication that vegetation is dead. 
    - Relevant for e.g. meadows of dead seagrass.
- **Age**: 
    - Indication of vegetation age. Used as a general numerical attribute. Exact meaning must be defined for each use case (e.g. age expressed as number of years old).
    - Relevant for young kelp in trawled areas 
- **Trawl track age**
- **Trawl track vegetation**
- **Turf underlying vegetation**
- **Periphyton**
    - Relevant for seagrass in Sandsund
- **Corralina officinalis abundance***
    - Ralavant for bedrock (Søla)
