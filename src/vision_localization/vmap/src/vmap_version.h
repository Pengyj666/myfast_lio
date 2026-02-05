#ifndef MOWER_VISION_LOCALIZATION_AS_VMAP_VERSION_H
#define MOWER_VISION_LOCALIZATION_AS_VMAP_VERSION_H

/* useful macro func */
#define STRING(S) #S
#define XSTRING(S) STRING(S)

/* version parsed out into numeric values */
#define NODE_VERSION_MAJOR 0
#define NODE_VERSION_MINOR 1
#define NODE_VERSION_PATCH 0

/* version as string */
#define NODE_VERSION \
  XSTRING(NODE_VERSION_MAJOR.NODE_VERSION_MINOR.NODE_VERSION_PATCH)
#define NODE_VERSION_DATE "2025.03.28 18:00 test"

/* name as string */
#define NODE_NAME "vmap_node"

#define COMPILE_TIME (__TIME__ " " __DATE__)

#endif  // MOWER_VISION_LOCALIZATION_AS_VMAP_VERSION_H
