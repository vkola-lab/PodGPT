#!/usr/bin/env python
# coding=utf-8
#
# MIT License
#
# Copyright (c) 2024 Kolachalama Lab at Boston University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
