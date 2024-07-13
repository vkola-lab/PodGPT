#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from answer_utils import extract_answer_for_spanish


completion = ("La respuesta correcta es A. La diplopía nunca puede ocurrir si se cubre un ojo."
              "\n\nLa diplopía es la percepción de dos imágenes de un objeto en lugar de una sola. "
              "Si se cubre un ojo, la imagen falsa desaparece, lo que significa que la diplopía se resuelve. "
              "Por lo tanto, la afirmación A es verdadera.\n\nLas demás opciones son incorrectas:\n\nB. "
              "La imagen exterior no siempre es la imagen falsa. La imagen exterior puede ser la verdadera "
              "y la interior la falsa, o viceversa.\n\nC. La parálisis del cuarto par craneal (par craneal "
              "temporal) ocurre cuando el paciente intenta mirar hacia arriba, hacia afuera y hacia adelante, "
              "no hacia arriba.\n\nD. La parálisis del sexto par craneal (par craneal accesorio superior) "
              "puede causar estrabismo convergente (no divergente), ya que el músculo lateral recto superior "
              "se encarga de mover el ojo hacia afuera y hacia arriba.")
completion = ("La mejor opción es C. 80 kJ/min.\n\nLa tasa de gasto de energía (E) se puede calcular utilizando "
              "la siguiente fórmula:\n\nE = VO2 x 21.14\n\nDonde VO2 es la tasa de consumo de oxígeno (en ml/min) "
              "y 21.14 es la energía liberada por la oxidación de 1 ml de oxígeno (en kJ).\n\nSi la tasa de consumo "
              "de oxígeno es de 2 l/min, podemos convertirla a ml/min multiplicando por 1000 (1 l = 1000 ml):\n\n"
              "VO2 = 2 l/min x 1000 ml/l = 2000 ml/min\n\nLuego, podemos calcular la tasa de gasto de energía:\n\n"
              "E = 2000 ml/min x 21.14 kJ/ml = 42.28 kJ/min\n\nRounded to the nearest whole number, "
              "the answer is approximately 80 kJ/min.")
completion = ("La respuesta es A.\n\nLas proteosomas son estructuras que se encargan de degradar "
              "proteínas en el retículo endoplásmico (ER).")
result = extract_answer_for_spanish(completion=completion)
print(result)
