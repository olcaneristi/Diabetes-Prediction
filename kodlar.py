## Kütüphaneleri çağırma ve yazdırma

import pandas as pd ## Veri analizi yapmak için kullandığımız Python dilinin yazılım kütüphanesi.
import matplotlib.pyplot as matplot ## Veri setimizin statik, analitik vb. görsel analizini yapmak için çağırıyoruz.
import seaborn as sea ## İstatistiksel veri görselleştirmesini gerçekleştirmesi için çağırıyoruz.

from sklearn.ensemble import RandomForestClassifier ## Rastgele Orman modellemesini kullanabilmek için çağırıyoruz.
from sklearn.model_selection import train_test_split ## Veri setimizi eğitim-test datası olarak bölmesi için çağırıyoruz.
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
### Doğruluk oranı, Sınıflandırma tablosu, hesaplaması için çağırıyoruz.

## Veri seti okutma ve veri keşfi

diabet_data = pd.read_csv('diabet.csv')
print(diabet_data)

diabet_data.info()

diabet_data.dropna(inplace=True)
statistics = diabet_data.describe()
print(statistics)

## Veri seti keşifsel grafik analizi

ax = sea.countplot(x="class" , data=diabet_data)
matplot.title('Diyabet konulan hasta sayısı (1- Evet, 0- Hayır)')
matplot.show()

## Kayıp/eksik veri analizi

miss_value = pd.isnull(diabet_data).sum()
print ("Kayıp/boş veri kontrolü: ")
print(miss_value)

## Dağılım grafiği tablosu

ax = sea.scatterplot(x="test", y="mass", hue="class", style="class", data=diabet_data)
matplot.show()

## Veriyi modellemeye hazırlama ve eğitim-test olarak bölme

veriX = diabet_data.iloc[:,0:8]
veriY = diabet_data.iloc[:,8]

x_egitim, x_test, y_egitim, y_test = train_test_split (veriX, veriY, test_size=0.24, random_state=42)
print("Bölmemizin ardından {} adet eğitim verisi ve {} adet test verimiz var.".format(x_egitim.shape[0], x_test.shape[0]))

## Veri modelleme ve tahmin sonuçları

rforest = RandomForestClassifier(n_estimators=300, random_state=42).fit(x_egitim, y_egitim.ravel())

randomf_prediction = rforest.predict(x_test)
randomf_prediction2= rforest.predict(x_egitim)

print("Eğitim başarı skorumuz: ", accuracy_score(y_egitim, randomf_prediction2))
print("Test başarı skorumuz: ", accuracy_score(y_test, randomf_prediction), '\n')
print("Sınıflandırma tablomuz aşağıdaki gibidir: \n", confusion_matrix(y_test, randomf_prediction))
print("Sınıflandırma raporu: \n ", classification_report(y_test, randomf_prediction))

## Sınıflandırma Tablosu Yorumlama

cm = confusion_matrix(y_test, randomf_prediction)
sea.heatmap(cm, center=True, annot=True, fmt="d", cmap='coolwarm')