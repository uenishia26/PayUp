import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("./formatedData.csv") 
print(df['label'].value_counts())

label_encoder = LabelEncoder() 
df['label']= label_encoder.fit_transform(df['label'])

x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123456, stratify=y)

smt = SMOTE(random_state=11)
X_train, y_train = smt.fit_resample(X_train, y_train)

print(df['label'].value_counts())
parameters = {'n_neighbors':np.arange(3,26, 2), 'weights':('uniform','distance'), 'algorithm':('auto','ball_tree','kd_tree','brute'), 'p':np.arange(1,11)}
knn = KNeighborsClassifier()
gscv = GridSearchCV(estimator=knn, param_grid=parameters, cv=5, scoring='recall_macro')
gscv.fit(X_train,y_train)
print("Hyperparameters:",gscv.best_params_)

y_pred = gscv.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=gscv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gscv.classes_)
disp.plot()
plt.show()

ocrtest = {'LA': [0.18574231523871812, 0.030085022890778287, 0.2302158273381295, 0.028122956180510136, 0.23217789404839764, 0.06998037933289732, 0.18770438194898625, 0.07194244604316546], 'Beverly': [0.3172007848266841, 0.07063440156965337, 0.46893394375408765, 0.06409417920209287, 0.4702419882275997, 0.10202746893394375, 0.3185088293001962, 0.10856769130150425], '-': [0.42576847612818836, 0.01896664486592544, 0.44408109875735774, 0.01831262262916939, 0.4460431654676259, 0.059516023544800525, 0.4277305428384565, 0.060170045781556575], '323-602-0637': [0.4676258992805755, 0.017004578155657292, 0.7253106605624591, 0.005232177894048398, 0.7272727272727273, 0.04708960104643558, 0.4695879659908437, 0.058862001308044476], '8480': [0.20994113799869196, 0.07521255722694571, 0.29561805101373445, 0.07128842380640942, 0.2969260954872466, 0.1092217135382603, 0.21124918247220406, 0.1131458469587966], 'Blvd': [0.4892086330935252, 0.06344015696533682, 0.5761935905820798, 0.059516023544800525, 0.5775016350555919, 0.0974493132766514, 0.49051667756703726, 0.1013734466971877], 'Ste': [0.5977763243950295, 0.058862001308044476, 0.6651406147809026, 0.05624591236102027, 0.6664486592544147, 0.09352517985611511, 0.5990843688685416, 0.09614126880313931], '1A': [0.6880313930673643, 0.05493786788750817, 0.7298888162197514, 0.052975801177240024, 0.7311968606932636, 0.09025506867233486, 0.6893394375408763, 0.09221713538260301], 'Los': [0.10333551340745585, 0.1209941137998692, 0.16808371484630477, 0.118378024852845, 0.17004578155657293, 0.16023544800523218, 0.105297580117724, 0.16285153695225638], 'Angeles': [0.18705035971223022, 0.11772400261608895, 0.34009156311314587, 0.11118378024852844, 0.34139960758665794, 0.15238718116415959, 0.18901242642249835, 0.15892740353172008], ',': [0.34532374100719426, 0.11118378024852844, 0.355134074558535, 0.1105297580117724, 0.3570961412688031, 0.15173315892740352, 0.34728580771746237, 0.15238718116415959], 'California': [0.3832570307390451, 0.1092217135382603, 0.5984303466317855, 0.1000654022236756, 0.5997383911052976, 0.1419228253760628, 0.38521909744931326, 0.1510791366906475], '90048-3414': [0.6206671026814912, 0.0987573577501635, 0.8384565075212557, 0.08960104643557881, 0.8397645519947678, 0.131458469587966, 0.6226291693917593, 0.1406147809025507], '05/30/2024': [0.25506867233485936, 0.15696533682145192, 0.4708960104643558, 0.14911706998037932, 0.4722040549378679, 0.18639633747547416, 0.2563767168083715, 0.19424460431654678], '06:53': [0.4918247220405494, 0.1484630477436233, 0.6023544800523217, 0.144538914323087, 0.6036625245258339, 0.18181818181818182, 0.49313276651406146, 0.18574231523871812], 'PM': [0.6213211249182472, 0.14388489208633093, 0.6671026814911707, 0.14257684761281883, 0.6684107259646828, 0.17920209287115763, 0.6226291693917593, 0.18116415958142576], 'GROCERY': [0.0013080444735120995, 0.395029431000654, 0.15173315892740352, 0.3924133420536298, 0.15238718116415959, 0.4264224983649444, 0.001962066710268149, 0.4290385873119686], '071100049': [0.0, 0.43623283191628515, 0.19424460431654678, 0.43100065402223675, 0.1948986265533028, 0.4669718770438195, 0.0013080444735120995, 0.4722040549378679], 'GOLDFISH': [0.2184434270765206, 0.4303466317854807, 0.3917593198168738, 0.42576847612818836, 0.3924133420536298, 0.4617396991497711, 0.2197514715500327, 0.46631785480706345], 'NF': [0.6736429038587312, 0.4185742315238718, 0.7161543492478745, 0.4172661870503597, 0.7174623937213865, 0.45454545454545453, 0.6749509483322433, 0.45585349901896666], '$': [0.7998691955526488, 0.9797253106605625, 0.8188358404185743, 0.9797253106605625, 0.8188358404185743, 0.999345977763244, 0.7998691955526488, 0.999345977763244], '8.99': [0.8109875735775016, 0.6285153695225638, 0.8979725310660562, 0.6265533028122956, 0.8986265533028123, 0.6671026814911707, 0.8116415958142577, 0.6690647482014388], '203220516': [0.0013080444735120995, 0.4780902550686723, 0.19555264879005888, 0.47416612164813604, 0.1962066710268149, 0.5107913669064749, 0.001962066710268149, 0.5147155003270111], 'SANPELL': [0.22040549378678875, 0.47351209941138, 0.37148463047743624, 0.4702419882275997, 0.3721386527141923, 0.5068672334859385, 0.2210595160235448, 0.5101373446697187], 'TF': [0.6769130150425114, 0.4617396991497711, 0.7181164159581426, 0.460431654676259, 0.7194244604316546, 0.49901896664486595, 0.6782210595160235, 0.5003270111183781], 'P': [0.7482014388489209, 0.7161543492478745, 0.7710922171353826, 0.7155003270111184, 0.7717462393721386, 0.7553956834532374, 0.7488554610856769, 0.7560497056899934], '8.59': [0.8051013734466972, 0.4578155657292348, 0.8927403531720078, 0.4551994767822106, 0.8940483976455199, 0.4944408109875736, 0.8064094179202093, 0.4970568999345978], 'Bottle': [0.11118378024852844, 0.5199476782210595, 0.2419882275997384, 0.5173315892740353, 0.24264224983649443, 0.5565729234793984, 0.1118378024852845, 0.5591890124264225], 'Deposit': [0.26357096141268804, 0.5173315892740353, 0.4159581425768476, 0.5147155003270111, 0.41661216481360364, 0.5533028122956181, 0.2642249836494441, 0.5559189012426422], 'Fee': [0.1988227599738391, 0.7769784172661871, 0.26880313930673644, 0.7737083060824068, 0.2707652060170046, 0.8083714846304775, 0.20078482668410727, 0.8116415958142577], '0.30': [0.7841726618705036, 0.5035971223021583, 0.8731196860693263, 0.5003270111183781, 0.8744277305428385, 0.5369522563767168, 0.7854807063440157, 0.540222367560497], 'HEALTH': [0.0013080444735120995, 0.5650752125572269, 0.131458469587966, 0.5624591236102028, 0.13211249182472204, 0.5984303466317855, 0.001962066710268149, 0.6010464355788097], 'AND': [0.15369522563767168, 0.5624591236102028, 0.2223675604970569, 0.5611510791366906, 0.22302158273381295, 0.5964682799215173, 0.15434924787442772, 0.5977763243950295], 'BEAUTY': [0.2432962720732505, 0.5604970568999346, 0.3734466971877044, 0.5578809679529104, 0.37410071942446044, 0.5938521909744932, 0.24395029431000653, 0.5964682799215173], '049060747': [0.0, 0.6075866579463701, 0.19686069326357097, 0.60431654676259, 0.197514715500327, 0.6383257030739045, 0.0006540222367560497, 0.6415958142576847], 'Gillette': [0.22040549378678875, 0.60431654676259, 0.39633747547416615, 0.6017004578155657, 0.3969914977109222, 0.6357096141268803, 0.2210595160235448, 0.6383257030739045], 'T': [0.06801831262262917, 0.9097449313276651, 0.08829300196206671, 0.9090909090909091, 0.08894702419882276, 0.94702419882276, 0.06867233485938522, 0.947678221059516], '36.99': [0.7880967952910399, 0.5873119686069327, 0.8966644865925442, 0.5840418574231524, 0.8979725310660562, 0.6232831916285154, 0.789404839764552, 0.6265533028122956], '049095924': [0.0, 0.6514061478090255, 0.19947678221059517, 0.6474820143884892, 0.2001308044473512, 0.6821451929365598, 0.0006540222367560497, 0.6860693263570962], 'Crest': [0.22171353826030085, 0.6468279921517331, 0.3302812295618051, 0.644865925441465, 0.33093525179856115, 0.6795291039895357, 0.2223675604970569, 0.6814911706998038], '3D': [0.3538260300850229, 0.644865925441465, 0.4002616088947024, 0.644211903204709, 0.40091563113145845, 0.6782210595160235, 0.3544800523217789, 0.6788750817527796], 'Whi': [0.4198822759973839, 0.6429038587311968, 0.47678221059516024, 0.6415958142576847, 0.4774362328319163, 0.6762589928057554, 0.42053629823413996, 0.6775670372792675], 'NON': [0.0, 0.6939175931981687, 0.06540222367560497, 0.6926095487246566, 0.06605624591236102, 0.7279267495094833, 0.0006540222367560497, 0.7292347939829954], 'RETAIL': [0.08960104643557881, 0.6919555264879006, 0.2223675604970569, 0.6886854153041203, 0.22302158273381295, 0.724002616088947, 0.09025506867233486, 0.7272727272727273], '004100019': [0.0, 0.737083060824068, 0.19816873773708307, 0.7325049051667757, 0.1988227599738391, 0.7691301504251145, 0.0006540222367560497, 0.7737083060824068], 'TARGET': [0.22302158273381295, 0.7325049051667757, 0.35186396337475473, 0.7298888162197514, 0.35251798561151076, 0.7658600392413342, 0.223675604970569, 0.7684761281883584], 'BAG': [0.37671680837148463, 0.7292347939829954, 0.4408109875735775, 0.7279267495094833, 0.44146500981033354, 0.7638979725310661, 0.37737083060824067, 0.7652060170045781], '0.00': [0.8122956180510137, 0.7148463047743623, 0.9012426422498365, 0.7128842380640942, 0.9018966644865926, 0.7527795945062132, 0.8129496402877698, 0.7547416612164813], 'Bag': [0.11118378024852844, 0.7809025506867233, 0.17854807063440156, 0.7776324395029431, 0.18051013734466972, 0.8129496402877698, 0.1131458469587966, 0.81621975147155], '0.10': [0.7926749509483323, 0.7606278613472858, 0.8803139306736429, 0.7586657946370177, 0.880967952910399, 0.7952910398953564, 0.7933289731850883, 0.7972531066056245], 'SUBTOTAL': [0.512753433616743, 0.854807063440157, 0.6860693263570962, 0.8508829300196207, 0.6867233485938522, 0.8862001308044474, 0.513407455853499, 0.8901242642249837], '63.96': [0.8188358404185743, 0.8456507521255723, 0.9274035317200785, 0.8436886854153042, 0.9280575539568345, 0.8803139306736429, 0.8194898626553303, 0.882275997383911], '=': [0.1157619359058208, 0.908436886854153, 0.1393067364290386, 0.907782864617397, 0.13996075866579463, 0.9457161543492478, 0.11641595814257685, 0.946370176586004], 'CA': [0.15761935905820798, 0.907128842380641, 0.20078482668410727, 0.9058207979071289, 0.2014388489208633, 0.9437540876389797, 0.15827338129496402, 0.9450621321124918], 'TAX': [0.22563767168083715, 0.9058207979071289, 0.2871157619359058, 0.9045127534336167, 0.28776978417266186, 0.9424460431654677, 0.2262916939175932, 0.9437540876389797], '9.50000': [0.31262262916939176, 0.9032047089601046, 0.46893394375408765, 0.8992805755395683, 0.4695879659908437, 0.9378678875081753, 0.3132766514061478, 0.9417920209287116], 'on': [0.4924787442773054, 0.8992805755395683, 0.5310660562459124, 0.8986265533028123, 0.5317200784826684, 0.9365598430346632, 0.49313276651406146, 0.9372138652714193], '54.87': [0.578155657292348, 0.8966644865925442, 0.6867233485938522, 0.8940483976455199, 0.6873773708306082, 0.9326357096141269, 0.578809679529104, 0.935251798561151], '5.21': [0.8397645519947678, 0.8894702419882276, 0.9280575539568345, 0.8881621975147155, 0.9287115761935906, 0.9254414650098103, 0.8404185742315239, 0.9267495094833225], 'TOTAL': [0.5814257684761281, 0.9404839764551994, 0.6886854153041203, 0.9391759319816874, 0.6893394375408763, 0.9744931327665141, 0.5820797907128843, 0.9758011772400261], '69.17': [0.8221059516023544, 0.9332897318508829, 0.9280575539568345, 0.9313276651406148, 0.9287115761935906, 0.9692609548724657, 0.8227599738391105, 0.9712230215827338], '60': [0.8207979071288424, 0.9790712884238064, 0.8639633747547416, 0.9784172661870504, 0.8639633747547416, 0.999345977763244, 0.8207979071288424, 0.999345977763244], '17': [0.8881621975147155, 0.9777632439502943, 0.9313276651406148, 0.9771092217135383, 0.9313276651406148, 0.999345977763244, 0.8881621975147155, 0.999345977763244]}

key_pred = {}
for key, val in ocrtest.items():
    ocr_pred = gscv.predict([val])
    key_pred[key] = ocr_pred
print(key_pred)