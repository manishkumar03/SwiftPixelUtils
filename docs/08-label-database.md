# Label Database: Complete ML Class Labels Reference

A comprehensive reference of class labels for popular machine learning models including ImageNet, COCO, Pascal VOC, Cityscapes, and more, with mappings and organizational strategies.

## Table of Contents

- [Introduction](#introduction)
- [ImageNet Labels](#imagenet-labels)
  - [ImageNet-1K (1000 Classes)](#imagenet-1k-1000-classes)
  - [ImageNet Hierarchy](#imagenet-hierarchy)
  - [ImageNet-21K](#imagenet-21k)
- [COCO Labels](#coco-labels)
  - [COCO 80 Classes](#coco-80-classes)
  - [COCO Category IDs](#coco-category-ids)
  - [COCO Supercategories](#coco-supercategories)
  - [COCO-Stuff](#coco-stuff)
- [Pascal VOC Labels](#pascal-voc-labels)
  - [VOC 20+1 Classes](#voc-201-classes)
  - [VOC Color Palette](#voc-color-palette)
- [Cityscapes Labels](#cityscapes-labels)
  - [Cityscapes 19 Classes](#cityscapes-19-classes)
  - [Cityscapes Full Labels](#cityscapes-full-labels)
- [ADE20K Labels](#ade20k-labels)
  - [ADE20K 150 Classes](#ade20k-150-classes)
  - [ADE20K-Full 847 Classes](#ade20k-full-847-classes)
- [Other Datasets](#other-datasets)
  - [Open Images (600 Classes)](#open-images-600-classes)
  - [LVIS (1203 Classes)](#lvis-1203-classes)
  - [Objects365 (365 Classes)](#objects365-365-classes)
  - [CIFAR-10/100](#cifar-10100)
  - [Fashion-MNIST](#fashion-mnist)
- [Action Recognition Labels](#action-recognition-labels)
  - [Kinetics-400](#kinetics-400)
  - [Kinetics-700](#kinetics-700)
- [Label Management](#label-management)
  - [Loading Labels](#loading-labels)
  - [Label Mapping](#label-mapping)
  - [Custom Label Sets](#custom-label-sets)
- [SwiftPixelUtils Label API](#swiftpixelutils-label-api)
- [Cross-Dataset Mapping](#cross-dataset-mapping)

---

## Introduction

Machine learning models output class indices that need to be mapped to human-readable labels. This reference provides complete label sets for major datasets and guidance on managing labels in your applications.

---

## ImageNet Labels

### ImageNet-1K (1000 Classes)

The standard ImageNet classification benchmark with 1000 categories:

```swift
/// ImageNet-1K class labels (indices 0-999)
let imagenetLabels: [String] = [
    // 0-9: Fish
    "tench",
    "goldfish", 
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "electric ray",
    "stingray",
    "rooster",
    "hen",
    "ostrich",
    
    // 10-19: Birds
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "American robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    
    // 20-29: Birds continued
    "American dipper",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "fire salamander",
    "smooth newt",
    "newt",
    "spotted salamander",
    "axolotl",
    
    // 30-39: Amphibians/Reptiles
    "American bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead sea turtle",
    "leatherback sea turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "green iguana",
    
    // 40-49: Reptiles
    "Carolina anole",
    "desert grassland whiptail lizard",
    "agama",
    "frilled lizard",
    "alligator lizard",
    "Gila monster",
    "European green lizard",
    "chameleon",
    "Komodo dragon",
    "Nile crocodile",
    
    // 50-59: Reptiles/Snakes
    "American alligator",
    "triceratops",
    "worm snake",
    "ring-necked snake",
    "eastern hog-nosed snake",
    "smooth green snake",
    "kingsnake",
    "garter snake",
    "water snake",
    "vine snake",
    
    // 60-69: Snakes continued
    "night snake",
    "boa constrictor",
    "African rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "Saharan horned viper",
    "eastern diamondback rattlesnake",
    "sidewinder",
    "trilobite",
    
    // 70-99: Invertebrates, Arachnids
    "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow",
    "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse",
    "peafowl", "quail", "partridge", "grey parrot", "macaw",
    "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater",
    "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose",
    
    // 100-199: More birds, mammals
    "black swan", "tusker", "echidna", "platypus",
    "wallaby", "koala", "wombat", "jellyfish", "sea anemone",
    "brain coral", "flatworm", "nematode", "conch", "snail",
    "slug", "sea slug", "chiton", "chambered nautilus",
    "Dungeness crab", "rock crab", "fiddler crab", "red king crab",
    "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "great egret", "bittern",
    "crane", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank",
    "dowitcher", "oystercatcher", "pelican", "king penguin",
    "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese Chin", "Maltese",
    "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon",
    "toy terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound",
    "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi",
    "Irish Wolfhound", "Italian Greyhound", "Whippet",
    "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
    "Scottish Deerhound", "Weimaraner",
    "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier",
    "Irish Terrier", "Norfolk Terrier", "Norwich Terrier",
    "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier",
    "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
    "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer",
    "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
    "West Highland White Terrier", "Lhasa Apso",
    
    // 200-299: Dogs, cats, animals
    "Flat-Coated Retriever", "Curly-coated Retriever",
    "Golden Retriever", "Labrador Retriever",
    "Chesapeake Bay Retriever", "German Shorthaired Pointer",
    "Vizsla", "English Setter", "Irish Setter",
    "Gordon Setter", "Brittany", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel",
    "Kuvasz", "Schipperke", "Groenendael", "Malinois",
    "Briard", "Australian Kelpie", "Komondor",
    "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres", "Rottweiler",
    "German Shepherd Dog", "Dobermann", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund",
    "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky",
    "Alaskan Malamute", "Siberian Husky", "Dalmatian",
    "Affenpinscher", "Basenji", "pug", "Leonberger",
    "Newfoundland", "Pyrenean Mountain Dog", "Samoyed",
    "Pomeranian", "Chow Chow", "Keeshond",
    "Griffon Bruxellois", "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog", "grey wolf",
    "Alaskan tundra wolf", "red wolf", "coyote", "dingo",
    "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat",
    "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah",
    
    // 300-399: Animals continued
    "brown bear", "American black bear", "polar bear",
    "sloth bear", "mongoose", "meerkat", "tiger beetle",
    "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly",
    "bee", "ant", "grasshopper", "cricket", "stick insect",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
    "dragonfly", "damselfly", "red admiral", "ringlet",
    "monarch butterfly", "small white", "sulphur butterfly",
    "gossamer-winged butterfly", "starfish", "sea urchin",
    "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver",
    "guinea pig", "common sorrel", "zebra", "pig", "wild boar",
    "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram", "bighorn sheep", "Alpine ibex", "hartebeest",
    "impala", "gazelle", "dromedary", "llama", "weasel",
    "mink", "European polecat", "black-footed ferret", "otter",
    "skunk", "badger", "armadillo", "three-toed sloth",
    "orangutan", "gorilla", "chimpanzee", "gibbon",
    "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey",
    "marmoset", "white-headed capuchin", "howler monkey",
    "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant",
    "African bush elephant", "red panda", "giant panda",
    
    // 400-499: Objects
    "barracouta", "eel", "coho salmon", "rock beauty",
    "clownfish", "sturgeon", "garfish", "lionfish", "pufferfish",
    "abacus", "abaya", "academic gown", "accordion", "acoustic guitar",
    "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron",
    "waste container", "assault rifle", "backpack", "bakery",
    "balance beam", "balloon", "ballpoint pen", "Band-Aid",
    "banjo", "baluster", "barbell", "barber chair", "barbershop",
    "barn", "barometer", "barrel", "wheelbarrow", "baseball",
    "basketball", "bassinet", "bassoon", "swimming cap",
    "bath towel", "bathtub", "station wagon", "lighthouse",
    "beaker", "military cap", "beer bottle", "beer glass",
    "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse",
    "bobsled", "bolo tie", "poke bonnet", "bookcase", "bookstore",
    "bottle cap", "bow", "bow tie", "brass", "bra", "breakwater",
    "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron",
    "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "tool kit", "carton", "car wheel",
    
    // 500-599: More objects
    "automated teller machine", "cassette", "cassette player",
    "castle", "catamaran", "CD player", "cello", "mobile phone",
    "chain", "chain-link fence", "chain mail", "chainsaw",
    "chest", "chiffonier", "chime", "china cabinet",
    "Christmas stocking", "church", "movie theater", "cleaver",
    "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "coil", "combination lock",
    "computer keyboard", "confectionery store", "container ship",
    "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "crane", "crash helmet", "crate",
    "infant bed", "Crock Pot", "croquet ball", "crutch",
    "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock",
    "digital watch", "dining table", "dishcloth",
    "dishwasher", "disc brake", "dock", "dog sled", "dome",
    "doormat", "drilling rig", "drum", "drumstick", "dumbbell",
    "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope",
    "espresso machine", "face powder", "feather boa",
    "filing cabinet", "fireboat", "fire engine", "fire screen sheet",
    "flagpole", "flute", "folding chair", "football helmet",
    "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat",
    "garbage truck", "gas mask", "gas pump", "goblet",
    
    // 600-699: More objects
    "go-kart", "golf ball", "golf cart", "gondola", "gong",
    "gown", "grand piano", "greenhouse", "grille", "grocery store",
    "guillotine", "barrette", "hair spray", "half-track",
    "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp",
    "harvester", "hatchet", "holster", "home theater",
    "honeycomb", "hook", "hoop skirt", "horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron",
    "jack-o'-lantern", "jeans", "jeep", "T-shirt", "jigsaw puzzle",
    "pulled rickshaw", "joystick", "kimono", "knee pad", "knot",
    "lab coat", "ladle", "lampshade", "laptop computer",
    "lawn mower", "lens cap", "paper knife", "library",
    "lifeboat", "lighter", "limousine", "ocean liner",
    "lipstick", "slip-on shoe", "lotion", "speaker",
    "loupe", "sawmill", "magnetic compass", "mail bag",
    "mailbox", "tights", "tank suit", "manhole cover",
    "maraca", "marimba", "mask", "match", "maypole",
    "maze", "measuring cup", "medicine chest", "megalith",
    "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten",
    "mixing bowl", "mobile home", "Model T", "modem",
    "monastery", "monitor", "moped", "mortar", "square academic cap",
    "mosque", "mosquito net", "scooter", "mountain bike",
    "tent", "computer mouse", "mousetrap", "moving van",
    "muzzle", "nail", "neck brace", "necklace", "nipple",
    "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "organ", "oscilloscope",
    "overskirt", "bullock cart", "oxygen mask", "packet",
    
    // 700-799: More objects
    "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute",
    "parallel bars", "park bench", "parking meter", "passenger car",
    "patio", "payphone", "pedestal", "pencil case", "pencil sharpener",
    "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel",
    "pirate ship", "pitcher", "hand plane", "planetarium",
    "plastic bag", "plate rack", "plow", "plunger",
    "Polaroid camera", "pole", "police van", "poncho",
    "billiard table", "soda bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile",
    "projector", "hockey puck", "punching bag", "purse",
    "quill", "quilt", "race car", "racket", "radiator",
    "radio", "radio telescope", "rain barrel", "recreational vehicle",
    "reel", "reflex camera", "refrigerator", "remote control",
    "restaurant", "revolver", "rifle", "rocking chair",
    "rotisserie", "eraser", "rugby ball", "ruler", "running shoe",
    "safe", "safety pin", "salt shaker", "sandal", "sarong",
    "saxophone", "scabbard", "weighing scale", "school bus",
    "schooner", "scoreboard", "CRT screen", "screw",
    "screwdriver", "seat belt", "sewing machine", "shield",
    "shoe store", "shoji", "shopping basket", "shopping cart",
    "shovel", "shower cap", "shower curtain", "ski", "ski mask",
    "sleeping bag", "slide rule", "sliding door", "slot machine",
    "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock", "solar thermal collector", "sombrero",
    "soup bowl", "space bar", "space heater", "space shuttle",
    
    // 800-899: More objects
    "spatula", "motorboat", "spider web", "spindle", "sports car",
    "spotlight", "stage", "steam locomotive", "through arch bridge",
    "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch",
    "stove", "strainer", "tram", "stretcher", "couch",
    "stupa", "submarine", "suit", "sundial", "sunglass",
    "sunglasses", "sunscreen", "suspension bridge", "mop",
    "sweatshirt", "swimsuit", "swing", "switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear",
    "television", "tennis ball", "thatched roof", "front curtain",
    "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole",
    "tow truck", "toy store", "tractor", "semi-trailer truck",
    "tray", "trench coat", "tricycle", "trimaran",
    "tripod", "triumphal arch", "trolleybus", "trombone",
    "tub", "turnstile", "typewriter keyboard", "umbrella",
    "unicycle", "upright piano", "vacuum cleaner", "vase",
    "vault", "velvet", "vending machine", "vestment", "viaduct",
    "violin", "volleyball", "waffle iron", "wall clock",
    "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower",
    "whiskey jug", "whistle", "wig", "window screen",
    "window shade", "Windsor tie", "wine bottle", "wing",
    "wok", "wooden spoon", "wool", "split-rail fence",
    "shipwreck", "yawl", "yurt", "website", "comic book",
    "crossword", "traffic sign", "traffic light", "dust jacket",
    "menu", "plate", "guacamole", "consomme", "hot pot",
    
    // 900-999: Food and misc
    "trifle", "ice cream", "ice pop", "baguette", "bagel",
    "pretzel", "cheeseburger", "hot dog", "mashed potato", "cabbage",
    "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana", "jackfruit",
    "custard apple", "pomegranate", "hay", "carbonara", "chocolate syrup",
    "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "cup", "eggnog", "alp",
    "bubble", "cliff", "coral reef", "geyser", "lakeshore",
    "promontory", "shoal", "seashore", "valley", "volcano",
    "baseball player", "bridegroom", "scuba diver", "rapeseed",
    "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric",
    "gyromitra", "stinkhorn mushroom", "earth star", "hen-of-the-woods",
    "bolete", "ear", "toilet paper"
]
```

### ImageNet Hierarchy

**Super-category organization:**

| Category | Index Range | Count |
|----------|-------------|-------|
| Animals | 0-397 | ~400 |
| Objects | 398-949 | ~550 |
| Food | 950-999 | ~50 |
| Scenes | 970-999 | ~30 |

```swift
// Group labels by super-category
let imagenetCategories: [String: ClosedRange<Int>] = [
    "fish": 0...6,
    "birds": 7...100,
    "reptiles": 30...68,
    "invertebrates": 69...100,
    "mammals": 101...299,
    "dogs": 151...268,
    "cats": 281...293,
    "primates": 364...384,
    "vehicles": 400...500,
    "household": 500...700,
    "food": 900...999
]
```

### ImageNet-21K

ImageNet-21K contains 21,841 categories. Here are the top-level synsets:

```swift
// ImageNet-21K includes all WordNet synsets
// Example subset structure:
let imagenet21kSample: [String: [String]] = [
    "animal": ["mammal", "bird", "fish", "reptile", "amphibian", "invertebrate"],
    "artifact": ["structure", "instrumentality", "commodity", "covering", "fabric"],
    "natural_object": ["plant", "fungus", "geological_formation", "body_of_water"],
    // ... ~21,000 total classes
]
```

---

## COCO Labels

### COCO 80 Classes

```swift
/// COCO 80-class labels for object detection
let cocoLabels: [String] = [
    "person",          // 0
    "bicycle",         // 1
    "car",             // 2
    "motorcycle",      // 3
    "airplane",        // 4
    "bus",             // 5
    "train",           // 6
    "truck",           // 7
    "boat",            // 8
    "traffic light",   // 9
    "fire hydrant",    // 10
    "stop sign",       // 11
    "parking meter",   // 12
    "bench",           // 13
    "bird",            // 14
    "cat",             // 15
    "dog",             // 16
    "horse",           // 17
    "sheep",           // 18
    "cow",             // 19
    "elephant",        // 20
    "bear",            // 21
    "zebra",           // 22
    "giraffe",         // 23
    "backpack",        // 24
    "umbrella",        // 25
    "handbag",         // 26
    "tie",             // 27
    "suitcase",        // 28
    "frisbee",         // 29
    "skis",            // 30
    "snowboard",       // 31
    "sports ball",     // 32
    "kite",            // 33
    "baseball bat",    // 34
    "baseball glove",  // 35
    "skateboard",      // 36
    "surfboard",       // 37
    "tennis racket",   // 38
    "bottle",          // 39
    "wine glass",      // 40
    "cup",             // 41
    "fork",            // 42
    "knife",           // 43
    "spoon",           // 44
    "bowl",            // 45
    "banana",          // 46
    "apple",           // 47
    "sandwich",        // 48
    "orange",          // 49
    "broccoli",        // 50
    "carrot",          // 51
    "hot dog",         // 52
    "pizza",           // 53
    "donut",           // 54
    "cake",            // 55
    "chair",           // 56
    "couch",           // 57
    "potted plant",    // 58
    "bed",             // 59
    "dining table",    // 60
    "toilet",          // 61
    "tv",              // 62
    "laptop",          // 63
    "mouse",           // 64
    "remote",          // 65
    "keyboard",        // 66
    "cell phone",      // 67
    "microwave",       // 68
    "oven",            // 69
    "toaster",         // 70
    "sink",            // 71
    "refrigerator",    // 72
    "book",            // 73
    "clock",           // 74
    "vase",            // 75
    "scissors",        // 76
    "teddy bear",      // 77
    "hair drier",      // 78
    "toothbrush"       // 79
]
```

### COCO Category IDs

**Important: COCO category IDs are NOT contiguous!**

```swift
/// Mapping from contiguous index (0-79) to COCO category ID
let cocoContiguousToCategory: [Int: Int] = [
    0: 1,    // person
    1: 2,    // bicycle
    2: 3,    // car
    3: 4,    // motorcycle
    4: 5,    // airplane
    5: 6,    // bus
    6: 7,    // train
    7: 8,    // truck
    8: 9,    // boat
    9: 10,   // traffic light
    10: 11,  // fire hydrant
    11: 13,  // stop sign (NOTE: 12 is missing!)
    12: 14,  // parking meter
    13: 15,  // bench
    14: 16,  // bird
    15: 17,  // cat
    16: 18,  // dog
    17: 19,  // horse
    18: 20,  // sheep
    19: 21,  // cow
    20: 22,  // elephant
    21: 23,  // bear
    22: 24,  // zebra
    23: 25,  // giraffe
    24: 27,  // backpack (NOTE: 26 is missing!)
    25: 28,  // umbrella
    26: 31,  // handbag (NOTE: 29-30 missing!)
    27: 32,  // tie
    28: 33,  // suitcase
    29: 34,  // frisbee
    30: 35,  // skis
    31: 36,  // snowboard
    32: 37,  // sports ball
    33: 38,  // kite
    34: 39,  // baseball bat
    35: 40,  // baseball glove
    36: 41,  // skateboard
    37: 42,  // surfboard
    38: 43,  // tennis racket
    39: 44,  // bottle
    40: 46,  // wine glass (NOTE: 45 missing!)
    41: 47,  // cup
    42: 48,  // fork
    43: 49,  // knife
    44: 50,  // spoon
    45: 51,  // bowl
    46: 52,  // banana
    47: 53,  // apple
    48: 54,  // sandwich
    49: 55,  // orange
    50: 56,  // broccoli
    51: 57,  // carrot
    52: 58,  // hot dog
    53: 59,  // pizza
    54: 60,  // donut
    55: 61,  // cake
    56: 62,  // chair
    57: 63,  // couch
    58: 64,  // potted plant
    59: 65,  // bed
    60: 67,  // dining table (NOTE: 66 missing!)
    61: 70,  // toilet (NOTE: 68-69 missing!)
    62: 72,  // tv (NOTE: 71 missing!)
    63: 73,  // laptop
    64: 74,  // mouse
    65: 75,  // remote
    66: 76,  // keyboard
    67: 77,  // cell phone
    68: 78,  // microwave
    69: 79,  // oven
    70: 80,  // toaster
    71: 81,  // sink
    72: 82,  // refrigerator
    73: 84,  // book (NOTE: 83 missing!)
    74: 85,  // clock
    75: 86,  // vase
    76: 87,  // scissors
    77: 88,  // teddy bear
    78: 89,  // hair drier
    79: 90   // toothbrush
]

/// Reverse mapping: COCO category ID to contiguous index
let cocoCategoryToContiguous: [Int: Int] = Dictionary(
    uniqueKeysWithValues: cocoContiguousToCategory.map { ($1, $0) }
)
```

### COCO Supercategories

```swift
/// COCO supercategory groupings
let cocoSupercategories: [String: [Int]] = [
    "person": [0],
    "vehicle": [1, 2, 3, 4, 5, 6, 7, 8],
    "outdoor": [9, 10, 11, 12, 13],
    "animal": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "accessory": [24, 25, 26, 27, 28],
    "sports": [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    "kitchen": [39, 40, 41, 42, 43, 44, 45],
    "food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "furniture": [56, 57, 58, 59, 60],
    "electronic": [62, 63, 64, 65, 66, 67],
    "appliance": [68, 69, 70, 71, 72],
    "indoor": [61, 73, 74, 75, 76, 77, 78, 79]
]
```

### COCO-Stuff

**COCO-Stuff adds 91 "stuff" classes to the 80 "thing" classes:**

```swift
let cocoStuffLabels: [String] = [
    // 0-79: Same as COCO 80 (things)
    // 80-90: More things (void in some versions)
    
    // 91-181: Stuff classes
    "banner", "blanket", "branch", "bridge", "building-other",
    "bush", "cabinet", "cage", "cardboard", "carpet",
    "ceiling-other", "ceiling-tile", "cloth", "clothes",
    "clouds", "counter", "cupboard", "curtain", "desk-stuff",
    "dirt", "door-stuff", "fence", "floor-marble", "floor-other",
    "floor-stone", "floor-tile", "floor-wood", "flower",
    "fog", "food-other", "fruit", "furniture-other", "grass",
    "gravel", "ground-other", "hill", "house", "leaves",
    "light", "mat", "metal", "mirror-stuff", "moss",
    "mountain", "mud", "napkin", "net", "paper",
    "pavement", "pillow", "plant-other", "plastic", "platform",
    "playingfield", "railing", "railroad", "river", "road",
    "rock", "roof", "rug", "salad", "sand",
    "sea", "shelf", "sky-other", "skyscraper", "snow",
    "solid-other", "stairs", "stone", "straw", "structural-other",
    "table", "tent", "textile-other", "towel", "tree",
    "vegetable", "wall-brick", "wall-concrete", "wall-other",
    "wall-panel", "wall-stone", "wall-tile", "wall-wood",
    "water-other", "waterdrops", "window-blind", "window-other",
    "wood"
]
```

---

## Pascal VOC Labels

### VOC 20+1 Classes

```swift
/// Pascal VOC segmentation labels (21 classes including background)
let vocLabels: [String] = [
    "background",    // 0
    "aeroplane",     // 1
    "bicycle",       // 2
    "bird",          // 3
    "boat",          // 4
    "bottle",        // 5
    "bus",           // 6
    "car",           // 7
    "cat",           // 8
    "chair",         // 9
    "cow",           // 10
    "diningtable",   // 11
    "dog",           // 12
    "horse",         // 13
    "motorbike",     // 14
    "person",        // 15
    "pottedplant",   // 16
    "sheep",         // 17
    "sofa",          // 18
    "train",         // 19
    "tvmonitor"      // 20
]
```

### VOC Color Palette

```swift
/// Official Pascal VOC color palette
let vocColorPalette: [(UInt8, UInt8, UInt8)] = [
    (0, 0, 0),         // 0: background - black
    (128, 0, 0),       // 1: aeroplane - maroon
    (0, 128, 0),       // 2: bicycle - green
    (128, 128, 0),     // 3: bird - olive
    (0, 0, 128),       // 4: boat - navy
    (128, 0, 128),     // 5: bottle - purple
    (0, 128, 128),     // 6: bus - teal
    (128, 128, 128),   // 7: car - gray
    (64, 0, 0),        // 8: cat - dark maroon
    (192, 0, 0),       // 9: chair - red
    (64, 128, 0),      // 10: cow - dark green
    (192, 128, 0),     // 11: diningtable - orange
    (64, 0, 128),      // 12: dog - dark purple
    (192, 0, 128),     // 13: horse - magenta
    (64, 128, 128),    // 14: motorbike - dark teal
    (192, 128, 128),   // 15: person - pink
    (0, 64, 0),        // 16: pottedplant - dark green
    (128, 64, 0),      // 17: sheep - brown
    (0, 192, 0),       // 18: sofa - bright green
    (128, 192, 0),     // 19: train - lime
    (0, 64, 128)       // 20: tvmonitor - dark blue
]

// Special class for boundary (optional, index 255)
let vocBoundaryColor = (224, 224, 192)  // Void/boundary
```

---

## Cityscapes Labels

### Cityscapes 19 Classes

```swift
/// Cityscapes evaluation labels (19 classes)
let cityscapesLabels: [String] = [
    "road",          // 0
    "sidewalk",      // 1
    "building",      // 2
    "wall",          // 3
    "fence",         // 4
    "pole",          // 5
    "traffic light", // 6
    "traffic sign",  // 7
    "vegetation",    // 8
    "terrain",       // 9
    "sky",           // 10
    "person",        // 11
    "rider",         // 12
    "car",           // 13
    "truck",         // 14
    "bus",           // 15
    "train",         // 16
    "motorcycle",    // 17
    "bicycle"        // 18
]

/// Cityscapes color palette
let cityscapesColorPalette: [(UInt8, UInt8, UInt8)] = [
    (128, 64, 128),    // 0: road - purple
    (244, 35, 232),    // 1: sidewalk - pink
    (70, 70, 70),      // 2: building - gray
    (102, 102, 156),   // 3: wall - blue-gray
    (190, 153, 153),   // 4: fence - pink-gray
    (153, 153, 153),   // 5: pole - gray
    (250, 170, 30),    // 6: traffic light - orange
    (220, 220, 0),     // 7: traffic sign - yellow
    (107, 142, 35),    // 8: vegetation - olive green
    (152, 251, 152),   // 9: terrain - light green
    (70, 130, 180),    // 10: sky - steel blue
    (220, 20, 60),     // 11: person - crimson
    (255, 0, 0),       // 12: rider - red
    (0, 0, 142),       // 13: car - dark blue
    (0, 0, 70),        // 14: truck - darker blue
    (0, 60, 100),      // 15: bus - dark cyan
    (0, 80, 100),      // 16: train - dark teal
    (0, 0, 230),       // 17: motorcycle - blue
    (119, 11, 32)      // 18: bicycle - dark red
]
```

### Cityscapes Full Labels

```swift
/// Cityscapes full label set (35 classes including ignored)
struct CityscapesFullLabel {
    let name: String
    let id: Int
    let trainId: Int  // -1 if ignored in training
    let category: String
    let hasInstances: Bool
    let color: (UInt8, UInt8, UInt8)
}

let cityscapesFullLabels: [CityscapesFullLabel] = [
    CityscapesFullLabel(name: "unlabeled", id: 0, trainId: -1, 
                        category: "void", hasInstances: false, 
                        color: (0, 0, 0)),
    CityscapesFullLabel(name: "ego vehicle", id: 1, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (0, 0, 0)),
    CityscapesFullLabel(name: "rectification border", id: 2, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (0, 0, 0)),
    CityscapesFullLabel(name: "out of roi", id: 3, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (0, 0, 0)),
    CityscapesFullLabel(name: "static", id: 4, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (0, 0, 0)),
    CityscapesFullLabel(name: "dynamic", id: 5, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (111, 74, 0)),
    CityscapesFullLabel(name: "ground", id: 6, trainId: -1,
                        category: "void", hasInstances: false,
                        color: (81, 0, 81)),
    CityscapesFullLabel(name: "road", id: 7, trainId: 0,
                        category: "flat", hasInstances: false,
                        color: (128, 64, 128)),
    CityscapesFullLabel(name: "sidewalk", id: 8, trainId: 1,
                        category: "flat", hasInstances: false,
                        color: (244, 35, 232)),
    CityscapesFullLabel(name: "parking", id: 9, trainId: -1,
                        category: "flat", hasInstances: false,
                        color: (250, 170, 160)),
    CityscapesFullLabel(name: "rail track", id: 10, trainId: -1,
                        category: "flat", hasInstances: false,
                        color: (230, 150, 140)),
    CityscapesFullLabel(name: "building", id: 11, trainId: 2,
                        category: "construction", hasInstances: false,
                        color: (70, 70, 70)),
    CityscapesFullLabel(name: "wall", id: 12, trainId: 3,
                        category: "construction", hasInstances: false,
                        color: (102, 102, 156)),
    CityscapesFullLabel(name: "fence", id: 13, trainId: 4,
                        category: "construction", hasInstances: false,
                        color: (190, 153, 153)),
    CityscapesFullLabel(name: "guard rail", id: 14, trainId: -1,
                        category: "construction", hasInstances: false,
                        color: (180, 165, 180)),
    CityscapesFullLabel(name: "bridge", id: 15, trainId: -1,
                        category: "construction", hasInstances: false,
                        color: (150, 100, 100)),
    CityscapesFullLabel(name: "tunnel", id: 16, trainId: -1,
                        category: "construction", hasInstances: false,
                        color: (150, 120, 90)),
    CityscapesFullLabel(name: "pole", id: 17, trainId: 5,
                        category: "object", hasInstances: false,
                        color: (153, 153, 153)),
    CityscapesFullLabel(name: "polegroup", id: 18, trainId: -1,
                        category: "object", hasInstances: false,
                        color: (153, 153, 153)),
    CityscapesFullLabel(name: "traffic light", id: 19, trainId: 6,
                        category: "object", hasInstances: false,
                        color: (250, 170, 30)),
    CityscapesFullLabel(name: "traffic sign", id: 20, trainId: 7,
                        category: "object", hasInstances: false,
                        color: (220, 220, 0)),
    CityscapesFullLabel(name: "vegetation", id: 21, trainId: 8,
                        category: "nature", hasInstances: false,
                        color: (107, 142, 35)),
    CityscapesFullLabel(name: "terrain", id: 22, trainId: 9,
                        category: "nature", hasInstances: false,
                        color: (152, 251, 152)),
    CityscapesFullLabel(name: "sky", id: 23, trainId: 10,
                        category: "sky", hasInstances: false,
                        color: (70, 130, 180)),
    CityscapesFullLabel(name: "person", id: 24, trainId: 11,
                        category: "human", hasInstances: true,
                        color: (220, 20, 60)),
    CityscapesFullLabel(name: "rider", id: 25, trainId: 12,
                        category: "human", hasInstances: true,
                        color: (255, 0, 0)),
    CityscapesFullLabel(name: "car", id: 26, trainId: 13,
                        category: "vehicle", hasInstances: true,
                        color: (0, 0, 142)),
    CityscapesFullLabel(name: "truck", id: 27, trainId: 14,
                        category: "vehicle", hasInstances: true,
                        color: (0, 0, 70)),
    CityscapesFullLabel(name: "bus", id: 28, trainId: 15,
                        category: "vehicle", hasInstances: true,
                        color: (0, 60, 100)),
    CityscapesFullLabel(name: "caravan", id: 29, trainId: -1,
                        category: "vehicle", hasInstances: true,
                        color: (0, 0, 90)),
    CityscapesFullLabel(name: "trailer", id: 30, trainId: -1,
                        category: "vehicle", hasInstances: true,
                        color: (0, 0, 110)),
    CityscapesFullLabel(name: "train", id: 31, trainId: 16,
                        category: "vehicle", hasInstances: true,
                        color: (0, 80, 100)),
    CityscapesFullLabel(name: "motorcycle", id: 32, trainId: 17,
                        category: "vehicle", hasInstances: true,
                        color: (0, 0, 230)),
    CityscapesFullLabel(name: "bicycle", id: 33, trainId: 18,
                        category: "vehicle", hasInstances: true,
                        color: (119, 11, 32)),
    CityscapesFullLabel(name: "license plate", id: -1, trainId: -1,
                        category: "vehicle", hasInstances: false,
                        color: (0, 0, 142))
]
```

---

## ADE20K Labels

### ADE20K 150 Classes

```swift
/// ADE20K scene parsing labels (150 classes)
let ade20kLabels: [String] = [
    "wall",            // 0
    "building",        // 1
    "sky",             // 2
    "floor",           // 3
    "tree",            // 4
    "ceiling",         // 5
    "road",            // 6
    "bed",             // 7
    "windowpane",      // 8
    "grass",           // 9
    "cabinet",         // 10
    "sidewalk",        // 11
    "person",          // 12
    "earth",           // 13
    "door",            // 14
    "table",           // 15
    "mountain",        // 16
    "plant",           // 17
    "curtain",         // 18
    "chair",           // 19
    "car",             // 20
    "water",           // 21
    "painting",        // 22
    "sofa",            // 23
    "shelf",           // 24
    "house",           // 25
    "sea",             // 26
    "mirror",          // 27
    "rug",             // 28
    "field",           // 29
    "armchair",        // 30
    "seat",            // 31
    "fence",           // 32
    "desk",            // 33
    "rock",            // 34
    "wardrobe",        // 35
    "lamp",            // 36
    "bathtub",         // 37
    "railing",         // 38
    "cushion",         // 39
    "base",            // 40
    "box",             // 41
    "column",          // 42
    "signboard",       // 43
    "chest of drawers", // 44
    "counter",         // 45
    "sand",            // 46
    "sink",            // 47
    "skyscraper",      // 48
    "fireplace",       // 49
    "refrigerator",    // 50
    "grandstand",      // 51
    "path",            // 52
    "stairs",          // 53
    "runway",          // 54
    "case",            // 55
    "pool table",      // 56
    "pillow",          // 57
    "screen door",     // 58
    "stairway",        // 59
    "river",           // 60
    "bridge",          // 61
    "bookcase",        // 62
    "blind",           // 63
    "coffee table",    // 64
    "toilet",          // 65
    "flower",          // 66
    "book",            // 67
    "hill",            // 68
    "bench",           // 69
    "countertop",      // 70
    "stove",           // 71
    "palm",            // 72
    "kitchen island",  // 73
    "computer",        // 74
    "swivel chair",    // 75
    "boat",            // 76
    "bar",             // 77
    "arcade machine",  // 78
    "hovel",           // 79
    "bus",             // 80
    "towel",           // 81
    "light",           // 82
    "truck",           // 83
    "tower",           // 84
    "chandelier",      // 85
    "awning",          // 86
    "streetlight",     // 87
    "booth",           // 88
    "television",      // 89
    "airplane",        // 90
    "dirt track",      // 91
    "apparel",         // 92
    "pole",            // 93
    "land",            // 94
    "bannister",       // 95
    "escalator",       // 96
    "ottoman",         // 97
    "bottle",          // 98
    "buffet",          // 99
    "poster",          // 100
    "stage",           // 101
    "van",             // 102
    "ship",            // 103
    "fountain",        // 104
    "conveyer belt",   // 105
    "canopy",          // 106
    "washer",          // 107
    "plaything",       // 108
    "swimming pool",   // 109
    "stool",           // 110
    "barrel",          // 111
    "basket",          // 112
    "waterfall",       // 113
    "tent",            // 114
    "bag",             // 115
    "minibike",        // 116
    "cradle",          // 117
    "oven",            // 118
    "ball",            // 119
    "food",            // 120
    "step",            // 121
    "tank",            // 122
    "trade name",      // 123
    "microwave",       // 124
    "pot",             // 125
    "animal",          // 126
    "bicycle",         // 127
    "lake",            // 128
    "dishwasher",      // 129
    "screen",          // 130
    "blanket",         // 131
    "sculpture",       // 132
    "hood",            // 133
    "sconce",          // 134
    "vase",            // 135
    "traffic light",   // 136
    "tray",            // 137
    "ashcan",          // 138
    "fan",             // 139
    "pier",            // 140
    "crt screen",      // 141
    "plate",           // 142
    "monitor",         // 143
    "bulletin board",  // 144
    "shower",          // 145
    "radiator",        // 146
    "glass",           // 147
    "clock",           // 148
    "flag"             // 149
]
```

### ADE20K-Full (847 Classes)

**ADE20K-Full extends the standard 150 classes to 847 classes for more fine-grained semantic segmentation:**

```swift
import SwiftPixelUtils

// Get ADE20K-Full labels
let label = LabelDatabase.getLabel(0, dataset: .ade20kFull)  // "wall"
let allLabels = LabelDatabase.getAllLabels(for: .ade20kFull)  // 847 labels

// For dense segmentation results
let segmentationResults = LabelDatabase.getTopLabels(
    scores: pixelProbabilities,
    dataset: .ade20kFull,
    k: 5
)
```

**Common models using ADE20K-Full:**
- Mask2Former
- OneFormer
- SegGPT
- Segment Anything (SAM) with semantic labels

---

## Other Datasets

### Open Images (600 Classes)

**Open Images V7 has 600 classes for object detection:**

```swift
import SwiftPixelUtils

// Get Open Images labels
let label = LabelDatabase.getLabel(0, dataset: .openimages)  // "Accordion"
let allLabels = LabelDatabase.getAllLabels(for: .openimages)  // 600 labels

// Top-K predictions
let topPredictions = LabelDatabase.getTopLabels(
    scores: modelOutput,
    dataset: .openimages,
    k: 5
)
```

**Common models using Open Images:**
- EfficientDet
- YOLO variants trained on Open Images
- RetinaNet

### LVIS (1203 Classes)

**LVIS (Large Vocabulary Instance Segmentation) v1 with 1203 long-tail classes:**

LVIS is designed to handle the long-tail distribution of real-world object categories, including rare classes that appear infrequently.

```swift
import SwiftPixelUtils

// Get LVIS labels
let label = LabelDatabase.getLabel(0, dataset: .lvis)  // "aerosol_can"
let allLabels = LabelDatabase.getAllLabels(for: .lvis)  // 1203 labels

// Top-K predictions
let topPredictions = LabelDatabase.getTopLabels(
    scores: modelOutput,
    dataset: .lvis,
    k: 10
)
```

**Common models using LVIS:**
- Mask R-CNN
- Cascade R-CNN
- DETR variants
- Federated loss models

### Objects365 (365 Classes)

**Objects365 v2 with 365 object detection classes:**

Objects365 is a large-scale object detection dataset with high-quality annotations, commonly used for pre-training detection models.

```swift
import SwiftPixelUtils

// Get Objects365 labels
let label = LabelDatabase.getLabel(0, dataset: .objects365)  // "Person"
let allLabels = LabelDatabase.getAllLabels(for: .objects365)  // 365 labels

// Top-K predictions
let topPredictions = LabelDatabase.getTopLabels(
    scores: modelOutput,
    dataset: .objects365,
    k: 5
)
```

**Common models using Objects365:**
- DINO
- Co-DETR
- Pre-trained detection backbones

### CIFAR-10/100

```swift
/// CIFAR-10 labels
let cifar10Labels: [String] = [
    "airplane",     // 0
    "automobile",   // 1
    "bird",         // 2
    "cat",          // 3
    "deer",         // 4
    "dog",          // 5
    "frog",         // 6
    "horse",        // 7
    "ship",         // 8
    "truck"         // 9
]

/// CIFAR-100 superclasses (20 groups)
let cifar100Superclasses: [String] = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2"
]
```

### Fashion-MNIST

```swift
/// Fashion-MNIST labels
let fashionMnistLabels: [String] = [
    "T-shirt/top",   // 0
    "Trouser",       // 1
    "Pullover",      // 2
    "Dress",         // 3
    "Coat",          // 4
    "Sandal",        // 5
    "Shirt",         // 6
    "Sneaker",       // 7
    "Bag",           // 8
    "Ankle boot"     // 9
]
```

---

## Action Recognition Labels

### Kinetics-400

**Kinetics-400 with 400 human action recognition classes:**

Kinetics-400 is a large-scale video dataset for human action recognition, containing 400 action classes.

```swift
import SwiftPixelUtils

// Get Kinetics-400 labels
let label = LabelDatabase.getLabel(0, dataset: .kinetics400)  // "abseiling"
let allLabels = LabelDatabase.getAllLabels(for: .kinetics400)  // 400 labels

// Top-K action predictions
let topActions = LabelDatabase.getTopLabels(
    scores: actionModelOutput,
    dataset: .kinetics400,
    k: 5
)

for action in topActions {
    print("\(action.label): \(String(format: "%.1f", action.confidence * 100))%")
}
```

**Common models using Kinetics-400:**
- I3D (Inflated 3D ConvNet)
- SlowFast Networks
- VideoMAE
- TimeSformer
- Video Swin Transformer

### Kinetics-700

**Kinetics-700 with 700 human action recognition classes:**

Kinetics-700 extends Kinetics-400 with 300 additional action classes for more comprehensive action recognition.

```swift
import SwiftPixelUtils

// Get Kinetics-700 labels
let label = LabelDatabase.getLabel(0, dataset: .kinetics700)  // "abseiling"
let allLabels = LabelDatabase.getAllLabels(for: .kinetics700)  // 700 labels

// Top-K action predictions
let topActions = LabelDatabase.getTopLabels(
    scores: actionModelOutput,
    dataset: .kinetics700,
    k: 10
)
```

**Common models using Kinetics-700:**
- TimeSformer
- Video Swin Transformer
- VideoMAE v2
- Omnivore

---

## Label Management

### Loading Labels

```swift
/// Label manager for loading and accessing class labels
class LabelManager {
    private var labels: [String] = []
    
    /// Load from bundled resource
    func loadFromBundle(name: String, extension ext: String = "txt") throws {
        guard let url = Bundle.main.url(forResource: name, 
                                        withExtension: ext) else {
            throw LabelError.fileNotFound
        }
        
        let content = try String(contentsOf: url, encoding: .utf8)
        labels = content.components(separatedBy: .newlines)
            .filter { !$0.isEmpty }
    }
    
    /// Load from JSON
    func loadFromJSON(name: String) throws {
        guard let url = Bundle.main.url(forResource: name, 
                                        withExtension: "json") else {
            throw LabelError.fileNotFound
        }
        
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String].self, from: data)
        labels = decoded
    }
    
    /// Load from JSON with index mapping
    func loadFromJSONWithMapping(name: String) throws {
        guard let url = Bundle.main.url(forResource: name,
                                        withExtension: "json") else {
            throw LabelError.fileNotFound
        }
        
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String: String].self, from: data)
        
        // Convert string keys to int indices
        let maxIndex = decoded.keys.compactMap { Int($0) }.max() ?? 0
        labels = [String](repeating: "unknown", count: maxIndex + 1)
        
        for (key, value) in decoded {
            if let index = Int(key) {
                labels[index] = value
            }
        }
    }
    
    /// Get label for index
    func label(for index: Int) -> String {
        guard index >= 0 && index < labels.count else {
            return "unknown (\(index))"
        }
        return labels[index]
    }
    
    /// Number of classes
    var count: Int { labels.count }
}
```

### Label Mapping

**Map between different label sets:**

```swift
/// Map COCO indices to VOC indices
let cocoToVoc: [Int: Int] = [
    1: 15,   // person → person
    2: 2,    // bicycle → bicycle
    3: 7,    // car → car
    4: 14,   // motorcycle → motorbike
    5: 1,    // airplane → aeroplane
    6: 6,    // bus → bus
    7: 19,   // train → train
    8: 4,    // boat → boat
    15: 8,   // cat → cat
    16: 12,  // dog → dog
    17: 13,  // horse → horse
    18: 17,  // sheep → sheep
    19: 10,  // cow → cow
    39: 5,   // bottle → bottle
    56: 9,   // chair → chair
    57: 18,  // couch → sofa
    58: 16,  // potted plant → pottedplant
    60: 11,  // dining table → diningtable
    62: 20   // tv → tvmonitor
]

/// Get VOC label from COCO detection
func vocLabelFromCoco(cocoIndex: Int) -> String? {
    guard let vocIndex = cocoToVoc[cocoIndex] else {
        return nil
    }
    return vocLabels[vocIndex]
}
```

### Custom Label Sets

```swift
/// Create custom label set for your model
struct CustomLabelSet {
    let labels: [String]
    let colors: [(UInt8, UInt8, UInt8)]?
    let parentMapping: [Int: String]?  // Supercategory mapping
    
    static func fromFile(_ filename: String) throws -> CustomLabelSet {
        // Load from config file
        let url = Bundle.main.url(forResource: filename, 
                                  withExtension: "json")!
        let data = try Data(contentsOf: url)
        
        struct Config: Decodable {
            let labels: [String]
            let colors: [[UInt8]]?
            let parentMapping: [String: String]?
        }
        
        let config = try JSONDecoder().decode(Config.self, from: data)
        
        let colors = config.colors?.map { 
            (UInt8($0[0]), UInt8($0[1]), UInt8($0[2]))
        }
        
        var parentMap: [Int: String]? = nil
        if let mapping = config.parentMapping {
            parentMap = [:]
            for (indexStr, parent) in mapping {
                if let index = Int(indexStr) {
                    parentMap?[index] = parent
                }
            }
        }
        
        return CustomLabelSet(
            labels: config.labels,
            colors: colors,
            parentMapping: parentMap
        )
    }
}
```

---

## SwiftPixelUtils Label API

```swift
import SwiftPixelUtils

// Built-in label sets
let imagenetLabels = Labels.imagenet1k
let cocoLabels = Labels.coco80
let vocLabels = Labels.pascalVOC

// Get label
let label = Labels.coco80[classIndex]

// Get color
let color = Labels.vocColor(for: classIndex)

// Load custom labels
try Labels.load(from: "custom_labels.json")

// Convenience methods
let result = classificationResult.topLabel  // Uses model's label set
let detections = detectionResult.withLabels(from: .coco80)
```

---

## Cross-Dataset Mapping

### Unified Class Mapping

```swift
/// Unified object categories across datasets
enum UnifiedClass: String, CaseIterable {
    case person
    case car
    case bicycle
    case motorcycle
    case bus
    case truck
    case boat
    case airplane
    case cat
    case dog
    case horse
    case cow
    case sheep
    case bird
    case chair
    case table
    case couch
    case tv
    case bottle
    case plant
}

/// Map to unified class from different datasets
struct UnifiedMapper {
    static func fromCoco(_ index: Int) -> UnifiedClass? {
        switch index {
        case 0: return .person
        case 1: return .bicycle
        case 2: return .car
        case 3: return .motorcycle
        case 4: return .airplane
        case 5: return .bus
        case 6: return nil  // train - no unified equivalent
        case 7: return .truck
        case 8: return .boat
        case 14: return .bird
        case 15: return .cat
        case 16: return .dog
        case 17: return .horse
        case 18: return .sheep
        case 19: return .cow
        case 39: return .bottle
        case 56: return .chair
        case 57: return .couch
        case 58: return .plant
        case 60: return .table
        case 62: return .tv
        default: return nil
        }
    }
    
    static func fromVoc(_ index: Int) -> UnifiedClass? {
        switch index {
        case 1: return .airplane
        case 2: return .bicycle
        case 3: return .bird
        case 4: return .boat
        case 5: return .bottle
        case 6: return .bus
        case 7: return .car
        case 8: return .cat
        case 9: return .chair
        case 10: return .cow
        case 11: return .table
        case 12: return .dog
        case 13: return .horse
        case 14: return .motorcycle
        case 15: return .person
        case 16: return .plant
        case 17: return .sheep
        case 18: return .couch
        case 19: return nil  // train
        case 20: return .tv
        default: return nil
        }
    }
}
```
