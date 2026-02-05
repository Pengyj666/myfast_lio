#ifndef MOWER_FUSION_LOCALIZATION_VERSION_H
#define MOWER_FUSION_LOCALIZATION_VERSION_H

/* useful macro func */
#define STRING(S) #S
#define XSTRING(S) STRING(S)

/* version parsed out into numeric values */
#define NODE_VERSION_MAJOR 0
#define NODE_VERSION_MINOR 0
#define NODE_VERSION_PATCH 1

/* version as string */
#define NODE_VERSION \
  XSTRING(NODE_VERSION_MAJOR.NODE_VERSION_MINOR.NODE_VERSION_PATCH)
#define NODE_VERSION_DATE "2025.10.30"

/* name as string */
#define NODE_NAME "fusion_localization_node"

#define COMPILE_TIME (__TIME__ " " __DATE__)

#endif  // MOWER_FUSION_LOCALIZATION_VERSION_H
