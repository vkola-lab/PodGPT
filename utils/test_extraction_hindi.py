#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from answer_utils import extract_answer_for_hindi

completion = ("  द. फोसिल ईंधन के प्रज्वलन (D. Fossil fuel burning) is not a source of anthropogenic carbon in the "
              "atmosphere. The correct answer is A. Respiration, C. Bacterial decomposition, and B. Photosynthesis "
              "are sources of anthropogenic carbon in the atmosphere. ")
completion = ("  ब. एक डीएनए सीक्वेंस जो प्रभाव के बिना जेनोम से हटा दिया जा सकता है. यह सटीकता रैट्रोट्रांस्पोज़ॉन का वर्णन करता है. "
              "यह एक डीएनए सीक्वेंस है जो गेन स्निप को हटा देता है और इसके बाद एक नया गेन स्निप ले आता है जो प्रभाव के बिना जेनोम से हटा दिया "
              "जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. "
              "इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बाद प्रभाव के बिना जेनोम से हटा दिया जा सकता है. इसके बा")
completion = (" जवाब D है, क्राईकोथाइरोइड मांसपेशी लंबे समय से बेहोश हो गई हैं लेकिन इंटेरर्टेनॉइड मांसपेशियां कमजोर हैं. "
              "यह कारण है कि क्राईकोथ मांसपेशी लंबे समय से बेहोश हो जाती है और इसलिए इसका लरिंगोस्कोपिक जांच अधिक करवाया जाता है. "
              "होना चाहिए कि इस प्रश्न के लिए क्राईकोथ मांसपेशी का लरिंगोस्कोपिक जांच क्यों कराया जाता है? "
              "क्योंकि इसका दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए "
              "कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए "
              "कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए "
              "कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए "
              "कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए "
              "कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी का दर बढ़ना चाहिए कि क्राईकोथ मांसपेशी ")
completion = "उत्तर: A. आर बैनिस्टर\n\nमाइल दौड़ के विश्व रिकॉर्ड 1886 में आर बैनिस्टर बनाया गया था।"
completion = (" a. अब्डुसेन्स है क्रेनियल नर्व सुगन्ध के संबंध में. यह एक स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है."
              " इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. "
              "इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है. इसका स्वाद स्वाद और सुगन्ध के लिए अपेक्षित रूप से अच्छा है."
              " इसका स्वाद स्वाद ")
completion = (" अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. "
              "'एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है।"
              " अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. "
              "एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ "
              "काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का "
              "मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. "
              "एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ "
              "काम करता है। अंतःस्रावी प्रणाली का मूल ग्रंथि होता है a. एड्रेनल के साथ पैंक्रियास के साथ काम करता है। अ"
              "ंतःस्रावी प्रणाली का मूल ग्रंथि होता है ए")
completion = ("सीधा जवाब: C. नेफ्रोलिथियासिस\n\nनेफ्रोलिथियासिस (Nephrolithiasis) का मतलब है कि गुर्दे में पथरी (renal calculus) "
              "की मौजूदगी।\n\nग्लोमेरुलोनेफ्राइटिस (A) एक प्रकार का नेफ्राइटिस है जिसमें ग्लोमेरुलस (glomeruli) क्षतिग्रस्त होते हैं, लेकिन पथरी की "
              "मौजूदगी से संबंध नहीं है।\n\nइंटरस्टीशियल नेफ्राइटिस (B) एक प्रकार का नेफ्राइटिस है जिसमें इंटरस्टीशियल सेल्स (interstitial cells) "
              "क्षतिग्रस्त होते हैं, लेकिन पथरी की मौजूदगी से संबंध नहीं है।\n\nपॉलीसिस्टिक किडनी (D) एक जन्मजात स्थिति है जिसमें गुर्दे में कई छोटे सिस्टर्स "
              "(cysts) होते हैं, लेकिन पथरी की मौजूदगी से संबंध नहीं है।")
completion = "सीधे सबसे अच्छे विकल्प के साथ जवाब देते हैं: B. मुख के इस्पात शिरों का लचीलापन और टेस्ट का नुकसान।"
result = extract_answer_for_hindi(completion=completion)
print(result)
