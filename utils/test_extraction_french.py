#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from answer_utils import extract_answer_for_french

completion = ("La bonne réponse est B. influence la position du viscérocrâne.\n\nLa synchondrose sphéno-occipitale est "
              "une synchondrose qui se développe entre l'os sphénoïde et l'os occipital, "
              "au niveau de la base du crâne. Elle est responsable de la croissance de la base "
              "du crâne et influence la position du viscérocrâne (c'est-à-dire la partie du crâne qui contient "
              "les organes cérébraux).\n\nLes autres options sont incorrectes :\n\nA. La synchondrose "
              "sphéno-occipitale est un cartilage de croissance primaire, qui est responsable de la "
              "croissance de la base du crâne pendant l'enfance et l'adolescence.\n\nC. La synchondrose "
              "sphéno-occipitale cesse son activité à l'âge de 14-16 ans, pas à 7 ans.\n\nD. "
              "La synchondrose sphéno-occipitale n'est pas réactivée chez les patients atteints "
              "d'acromégalie, qui est une maladie caractérisée par une hyperplasie de la glande pituitaire.")
completion = ("La bonne réponse est A. Vaccination antitétanique.\n\nLa prévention primaire vise à éviter "
              "la survenue d'une maladie ou d'un problème de santé en prévenant la contamination ou "
              "l'exposition à un agent pathogène. La vaccination antitétanique est un exemple de prévention "
              "primaire, car elle permet de prévenir la maladie du tétanos en inoculant un vaccin qui stimule "
              "l'immunité du corps contre le tétanos.\n\nLes autres options ne sont pas des exemples de prévention "
              "primaire :\n\n* B. Dépistage du cancer colorectal par l'Hémoccult (recherche d'Hémoglobine dans "
              "les selles) : c'est un exemple de prévention secondaire, car il vise à détecter une maladie déjà "
              "présente.\n* C. Prévention des récidives d'accident vasculaire cérébral par les anti-agrégants : "
              "c'est un exemple de prévention secondaire, car il vise à prévenir la récidive d'une maladie déjà "
              "survenue.\n* D. Rééducation des coronariens : c'est un exemple de prévention tertiaire, car "
              "il vise à réduire les conséquences d'une maladie déjà survenue.\n* E. Réinsertion sociale des "
              "handicapés : c'est un exemple de prévention tertiaire, car il vise à améliorer la qualité de "
              "vie des personnes handicapées, mais ne vise pas à prévenir la survenue d'une maladie.")
completion = ("La réponse est C : l'acétylcholine. L'acétylcholine est un messager "
              "extracellulaire et non un messager intracellulaire.")
completion = ("La réponse est la C : l'épiderme. L'épiderme est la couche la plus externe d'une feuille "
              "et est la première à recevoir la lumière du soleil. C'est là que se déroule la "
              "photosynthèse et la production d'oxygène.")
completion = "La réponse est C : séquence.\n\nLa hypoplasie pulmonaire est une séquence."
completion = ("La réponse est D : le syndrome de Waardenburg. Le syndrome de Waardenburg "
              "n'est pas causé par une mutation dans FGFR3.")
completion = ("Réponse: E. C5a\n\nC5a est une molécule issue de l'activation du complément "
              "qui possède une activité chimiotactique.")
completion = ("La spécificité d'un test est la probabilité que le test soit positif lorsque "
              "le résultat réel est négatif.\n\nLa spécificité de l'examen est de 0,71, ce qui signifie "
              "que 71 % des malades négatifs auront un résultat positif.")
completion = ("La bonne réponse est **B. Tinidazole (FASIGYNE®)**.\n\nTinidazole est un médicament "
              "de première ligne pour traiter les infections intestinales aiguës.")
completion = ("L'option D. Sel disodique monocalcique de l'EDTA est "
              "un antidote utilisé pour l'intoxication par le plomb.")
completion = ("La sensibilité de d'examen est de 0,71.\n\nLa sensibilité est la probabilité qu'un test "
              "sera positif lorsque le résultat réel est positif.")
completion = ("La réponse A est la plus juste.\n\nLa prévalence se définit comme le "
              "nombre de nouveaux cas pendant une certaine période, multiplié par la durée moyenne de la maladie.")
completion = ("**A. Met en oeuvre la mesure de l'intensité lumineuse d'un rayonnement polychromatique**\n\n"
              "La mesure de l'intensité lumineuse d'un rayonnement polychromatique est utilisée pour "
              "effectuer un dosage par photométrie de flamme.")
completion = ("La réponse est **A. Sa clairance soit proportionnelle à la dose**\n\nLa clairance est "
              "proportionnelle à la dose. Cela signifie que la clairance augmentera avec une augmentation de la dose.")
completion = ("La forme éliminée de façon prépondérante par voie urinaire est l'urée. "
              "L'urée est la forme d'azote organique qui est éliminée de l'organisme sous forme d'urine.")
completion = (" La réponse correcte est C. Il peut être mortel. Les autres options sont les suivantes :\nA. "
              "Il n'est présent qu'en Afrique sub-saharienne. Incorrecto. El paludismo es una enfermedad transmitida "
              "por vectores que se encuentra en muchas partes del mundo, no solo en África.\nB. Las recaídas son "
              "causadas por los hipnozoitos. Correcto. Los hipnozoitos son formas latentes de los parásitos que "
              "pueden causar recaídas de la enfermedad después de que se ha tratado.\nD. Se declara habitualmente "
              "3 años después del retorno de zona de endemia. Incorrecto. La incubación del paludismo es de 10 a "
              "15 días después de la picadura del mosquito vector.\nE. Se trata con fluconazol. Incorrecto. "
              "El tratamiento del paludismo es con antimaláricos, no con fluconazol.")
completion = (" La réponse est D. Le polyéthylène glycol (PEG) 400 est le plus utilisé en compression. En effet, "
              "le PEG 400 est utilisé en granulation, non en compression. Les excipients de compression, tels que "
              "les diluants, lubrifiants, liants, etc., ne comprennent pas le PEG 400.")
result = extract_answer_for_french(completion=completion)
print(result)
