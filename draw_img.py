# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:26:47 2022

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    rew_to_dist = lambda x: np.log(x + 1/2)*(-3)
    
    a2c_test_rew = [0.5, 0.5, 0.5, 0.4973992212338233, 0.4933555064030559, 0.493665978211411, 0.49056561935607834, 0.4708499709739258, 0.4453027797129855, 0.43346280348255006, 0.47095110635016135, 0.45619667157275057, 0.4455720247759746, 0.49309041241089135, 0.39003827833563254, 0.29173549043345115, 0.24742301888407658, 0.24872178909088505, 0.2984016084597516, 0.42283166022891117, 0.4823405755088822, 0.4828491829879822, 0.4247763162431005, 0.3409100555250658, 0.25956057255607334, 0.1824710274286102, 0.21358971014558625, 0.26118073769960615, 0.3040372167326718, 0.32734336104275785, 0.29572311104656124, 0.21222754307207936, 0.12152223490212366, 0.10160033050108619, 0.11911185132068347, 0.1595225448035219, 0.22724071665478618, 0.28740987525884976, 0.2925525928225957, 0.3504639246449913, 0.40058478114873597, 0.4814715042807093, 0.4938754904657845, 0.45001685765211763, 0.35070638315949487, 0.21210108656323035, 0.2759438426762867, 0.16573966543970253, 0.0701794106781316, 0.012350847165820578, -0.01206318856970584, 2.751877224949073e-05, 0.13846523095073104, 0.3412966936029326, 0.3727589373842809, 0.12364228547551626, -0.06257581539404644, -0.18176049501646074, -0.26313995173093613, -0.3099037621161874, -0.33723413832669247, -0.3495961467140862, -0.352928468690867, -0.34822488031928667, -0.3584051787369599, -0.34739597143088036, -0.3231647999393259, -0.28319108663151216, -0.2278167490405077, -0.1573896778228081, -0.08965693496924465, -0.026633443914146293, 0.019492679484412356, 0.06727640492542508, 0.2730629278980091, 0.3274456065138247, 0.42690934848401396, 0.4680750381177379, 0.42737001611522274, 0.42047676686199786, 0.40341879483997434, 0.3235030209415657, 0.17108344811364595, 0.11509361481763658, 0.10519219714369166, 0.13046105212035952, 0.18629214284229734, 0.24063928982015392, 0.24226776739806588, 0.22531275407970686, 0.24070440566354756, 0.3005074252540336, 0.3685827726614679, 0.4132720249413495, 0.42546634421336815, 0.3708824826240973, 0.30981987218071105, 0.23946090018929322, 0.18009522113043797, 0.15867823537005699, 0.1734631271315722, 0.11578943837185285, 0.13342498605394904, 0.2173548799622097, 0.3023413963298093, 0.43063718680554197, 0.41569561815725264, 0.3280312840600722, 0.29795401978371094, 0.43438309225347105, 0.4115831361901532, 0.32187144835484804, 0.29988141198785156, 0.2856052430705034, 0.2556413478092786, 0.17237055248639066, 0.17390599234006943, 0.2252765809902364, 0.3314475912779363, 0.49960138201006865, 0.3272895629503503, 0.2258996372987555, 0.29191145081189684, 0.3889690175774275, 0.4200221613957139, 0.22793736138179643, 0.08895565177929243, 0.007137572453498442, -0.04401282592455369, -0.06994941455015385, -0.10874390297893932, -0.12058359943705471, -0.10448533604446414, -0.06386866872969116, -0.009517190163990774, 0.07216187132171492, 0.2182542395865259, 0.39747506940902255, 0.4941653814902549, 0.48117128367087436, 0.4095077982992904, 0.3315922678051203, 0.30541407562863565, 0.331607859286401, 0.41622278098271936, 0.47249805130182887, 0.4345724178076401, 0.45062906766874755, 0.4767578954095265, 0.34874687101939095, 0.21194362625086793, 0.12657471683059807, 0.09764388994035023, 0.057845182472227474, 0.07071177457317601, 0.07645259363664814, 0.12501516949224334, 0.24277979462670218, 0.39751367629744316, 0.49368428702030054, 0.4617206310566363, 0.45145409759401245, 0.49675097950602576, 0.4168540804092292, 0.3371427531745671, 0.3501817711952431, 0.33054022333517286, 0.33362266610479396, 0.3351386883409563, 0.31695202120793853, 0.2546670731711478, 0.19482277214857524, 0.12070976563791314, 0.09704809618397181, 0.14595473635053569, 0.3004112214604122, 0.42591288780037395, 0.47681266998521576, 0.4383470841128433, 0.2355954008700567, 0.12666302351172554, 0.06402936259151693, 0.029736489631290164, 0.02616715270999337, -0.04780844507055115, -0.053651426827731885, -0.031695742606224486, 0.01914948616746337, 0.1125343457281528, 0.2809716809884637, 0.43958772353324194, 0.15919117973773733, 0.08028431326461072, -0.07069833543196152, -0.17578650905573223, -0.23925184257998028, -0.2809533957888335, -0.30385818945671395, -0.31240201507882165, -0.30904653522483455, -0.2775891774074215, -0.22027765611521516, -0.1640839180072715, -0.12113553506703306, -0.10016125445209734, -0.0885740594968743, -0.04740052774073594, -0.05140432597025424, 0.08103578041741666, 0.2121432992219141, 0.32298339629193695, 0.3933866893534207, 0.4126774793590694, 0.3735178141681287, 0.25482072206458417, 0.1968394982837749, 0.11180891696975492, 0.010944059987285759]
    a2c_test_dist = [rew_to_dist(x) for x in a2c_test_rew]
    
    a2c_rews = [0,12.348325285932628,3.723903634110569,7.267197048571427,
                4.760962593916039,4.177981510668443,7.418025828463595,
                3.6209531855591663,3.5523187606391176,5.366135195566825,
                5.280113003333306,8.027357838960285,4.859718179279041,
                4.409843252783985,3.8678871179586016,4.648145089802616,
                9.392766759178405,9.156016610067406,3.2962656667095374,
                8.32077172677118,3.448308764416013,4.556764301884593,
                5.514646401416404,5.947944173175247,3.555667464728825,
                5.696953549922561,4.330873416162932,3.805920606910933,
                3.4034078012966478,4.825958307902973,9.361321674772114,
                8.283900187050252,4.811910345316843,3.8338515156707187,
                5.168736066902877,4.409025063409761,11.474538077308736,
                4.3778560609375194,6.380664710397054,7.6443629106596696,
                4.674445673366328,7.0943637209515975,4.272200141136241,
                6.255045895086279,8.221007168534335,2.6590564401168835,
                4.655984098046625,4.221946185396448,4.65827021173315,
                3.9560953823300165,11.229115674371231,9.590592115971987,
                4.34011760978519,4.2076431600951985,3.9611949012484846,
                5.190526607366581,4.225775023790777,6.12028566894667,
                6.566503530712379,4.815240656502036,8.168184574539275,
                3.0779041975540276,5.917069953312935,4.874958275430046,6.453525494307749,7.164736819259923,3.9273049204191715,6.991855862547646,3.8019006059510483,4.709810133486697,4.459356911403434,4.175415796048512,4.793689728892533,4.985570859588357,13.895301621045087,3.997098116736489,8.407102701991374,10.48912924813861,6.63131913563976,15.398241334472672,8.254173858595497,17.990528364427057,15.123405246873094,5.4191137878812485,5.176085890837021,5.313296273087868,9.432182424872028,5.20944845102837,8.174341566492267,23.793695804651854,5.035296706017716,13.071190548593597,15.426680499168615,4.450734338716178,4.310729846108006,5.309888567737763,17.26511274298402,20.12347907972912,5.058034879512028,3.412708787976282,4.273880751318413,8.874227727859932,3.9836541535740952,4.87441955541218,9.553044967436307,6.283623922876382,9.546481951138048,16.295905237775845,13.565365043957655,9.118375316965588,13.285809198210353,5.535835626614552,10.833397082491553,7.479943057225077,10.086920877925486,7.248810184391958,8.942731057809832,11.291154477674239,6.1765633404001665,12.989371711485866,14.578482118251511,13.415225238572004,10.451153609977977,8.676185592592976,9.276875307734889,9.063958272634013,28.337664506977884,20.525482483822653,25.19371537923287,29.72020187699409,11.209872973842531,19.138772440986372,24.705973341291728,9.015687277877536,31.426658766815226,53.254966626125565,18.260125840119446,14.662285394782016,26.196176998131516,10.68090952566534,39.53383034493044,36.08124594251294,25.589954683087477,23.827001346684686,12.258976208816415,11.14345723170399,11.255517271525099,14.154984151186772,8.444216022798814,29.433174948945215,10.96078450441217,44.34688396526357,84.02272818901916,38.65226181367905,18.273054849296404,7.547565020321647,13.75791934690862,13.607361160030532,16.83718321844139,6.518423611139654,10.683319294456567,9.008829329884763,19.66470307471269,8.738502641376826,4.942269718366051,5.508939041163023,5.852960202979792,10.384616270426738,7.356668983670786,7.464012817480045,17.460247630278847,8.63533379019158,12.538570631037828,7.058289448738894,8.018190928664097,19.980279221310724,12.076733675947516,35.07561623125159,45.47283162625435,36.840740214450214,18.667214694310438,11.118604633914218,10.80913987438258,20.99018734539534,17.952776962691473,5.114887681999876,3.467827254454627,6.095283134038634,20.031313780436175,14.672534207326416,14.057026127221953,21.70773329141791,12.076154620934757,9.289497703987113,11.613308973618983,17.590762488818203,28.99801330986655,51.242604426043506,13.218221525891437,19.77170974897989,80.02938719365508,49.01408327698695,40.02540528656139,11.938994207811781,40.230765970216126,14.487994561245264,56.66838461050491,21.560571035635842,14.659751161703282,8.552349305576927,16.861272941145447,21.62560730697168,12.038097653737044,10.297433331961003,9.857485049054379,25.810316793729562,17.563982905597296,25.167401559784544,8.976895278391385,23.011188582034848,17.542905527512733,9.535651965804947,9.648279733616114,45.30791359628169,16.916762389000944,9.148823362679064,12.579567002374068,48.90122941523111,8.96871759612495,25.910190904652637,15.853070770141606,45.44876656013571,12.44584600567077,12.56114987310272,21.589576101194027,11.910616346444757,9.964554615033292,7.786859281774881,10.508922996731778,8.943037292829644,10.851467538080602,18.469033015154324,11.125126443610386,8.191997375805242,7.793328514685884,7.2290768680773105,7.863993503125772,12.216706029958118,9.564539548093398,30.214330572907812,11.39969849429703,60.9094094741127,32.40749672665356,60.32626104013386,18.240872409780867,6.577436998142884,8.416422379603848,33.02683794622554,15.379416746269632,33.9152971503328,74.95280595560958,57.32014290357396,40.08854930886985,51.61731895436435,44.79528727143638,13.006809821170526,37.16338109661785,24.578400610979923,18.30636850133918,37.05019598688589,22.52637672980297,37.82299048103755,37.46030445178731,40.97304364052351,36.666942293918154,71.29758892213063,45.261859919383106,54.33040237138986,14.667208714717743,13.609113025026923,13.674386687614387,12.769481592132719,30.340722755519455,49.35216065303932,14.148352877134517,13.718313170838792,46.51156831079989,39.821864095689776,60.290700679163436,55.12769167302669,67.6137474335664,71.63924895610887,52.21244307561928,62.701329504937995,8.22482640667433,48.72244198087265,53.48663864480674,27.131823209132815,12.829461383085489,66.54046226544429,92.59727279304376,71.19245891960202,10.183975176683182,7.0750028432387735,15.704555143878613,18.50641155214591,13.959470579967592,30.558974088463163,9.797325393541593,11.962845417025806,40.83823009309774,10.38567861412092,7.569897736302835,7.965276199066979,37.35906032942224,35.825821369617785,89.74685417941959,38.500698745132084,33.20480638998374,18.537872869678186,20.568052137941443,13.025310074614739,34.343870267294704,30.739401830467404,13.182556186734626,68.17630847630426,74.83873326635205,34.06019172486063,38.09146008560091,30.29512581988556,19.677820752174704,13.781984056684792,82.76685947366163,62.14249811935987,39.61408745386179,38.206324734958656,24.697009273682443,11.662278370386728,73.62236157419994,68.05607847003891,68.79117627995336,21.545750044164475,15.781264156468415,14.214236740803884,23.143559595657248,24.808853262471636,19.92119572437126,39.84299485719656,9.145182730321617,6.808362045945179,7.513324304140404,23.944008317070562,56.68712595209674,9.516297243677244,37.599868604114164,7.940819443240695,9.579346088015681,7.225152874870238,19.448332272359835,8.373532883641708,20.726268252735252,23.41108710456919,7.479874547174566,24.89412074661285,18.22155974553917,9.164203308624428,8.88738242964377,52.154901039216966,7.970624302598099,7.004904909514009,61.67332789643782,40.06689776214246,31.18672717564547,16.808881077422992,13.045514400487534,13.196967546220652,7.447376747301901,16.85850874311259,15.298726204708286,9.399477652246347,18.00561008010874,9.809664566859524,14.44621562550708,9.412640471831448,9.933556647461574,8.70963603574043,11.440460670305026,10.017461790393346,10.102010090475085,9.587060069574015,8.756068316401178,12.074556128392603,10.52685906591191,14.884129684928459,17.983424243942554,76.36317169186819,13.114470524725942,10.504056380314625,11.323952000769443,36.80449002717336,10.890630503943472,15.761779128233483,57.07847704242282,9.736833527610493,33.87761810237482,9.938881604940832,61.849829464186485,82.11902137683704,36.816312680160515,38.67920707668609,12.242264822903314,28.71394683836149,32.16387310466495,54.57013372313811,90.9303567647824,39.887345660355216,52.28917988272776,32.119224719128255,10.940393737951359,11.496369657037976,37.20822932317393,41.67443745078027,65.02179646604581,8.372300569483697,12.327672606542258,11.291891902348445,48.43522542457684,9.501351781427232,61.71506467504058,35.826945701311764,33.5721317342569,40.518410898129545,75.77728140965857,11.858697271539047,48.35522516417369,10.387349180593416,9.175331135539516,36.551719630768844,20.2141942178544,11.805205449898768,62.585152848982815,58.687170292014876,26.624842831789007,11.603907391257438,51.47959330818058,11.376609802212744,33.67027478281075,45.137881843228556,89.66507577765803,57.324207578358454,29.796699164287382,54.645064800150635,16.38944541131669,7.731922150195297,12.204417895303763,11.952888625653658,12.80624958246067,14.614589558414492,15.838444611802782,11.555056699176781,12.845741203030757,17.95111186674473,15.498456751313144,10.15714557222842,28.40625103045085,29.285661861915067,9.890778738652136,9.176705786204286,16.258163688854275,36.76550437550159,28.23106370879178,59.961808091023585,32.279595376300904,68.13644745788882,43.27510844664426,53.39925451834499,32.23985231179588,6.446846127162313,16.17167878095188,11.422668479234455,13.093011855945674,24.92129348513269,7.574481433684201,6.332690809063891,9.952398367333531,9.155479457228022,9.857578154372879,12.986193511964984,10.5610477768105,37.172212103289986,47.13775257562078,56.43952397096057,78.4778203195367,60.44808183786707,11.28283473751127,32.687414153122695,13.066739350296299,103.72559962626106,26.84286965781159,40.043912626134905,7.3072509698962,10.5112248641044,46.26436207774473,46.96768512692984,62.542036766017745,49.868229258416186,46.08417923973804,36.68011077140421,61.97081916280818,19.809080545469573,12.486504067254229,12.380079567324096,47.542438948373196,64.02142938829688,79.6614709258306,69.9250191348892,37.68269795483189,36.34374613888154,16.559560598183737,8.387805860350278,9.761191205854287,73.7953476234017,75.38680075145538,90.50574835467386,36.04704120543275,14.954233400392495,72.31317793131738,40.9994862185404,31.289180875646835,83.3006114080372,65.47174191579133,39.215331118951525,105.46340824338493,43.32177478193266,46.649093383300595,9.481694079575572,117.50081858888554,17.58641510843441,79.94594491989342,108.10800574548773,19.785141346743945,113.35178213545703,90.06718957630784,97.94156019306271,94.00717376102476,9.444638782501427,14.101198183516525,24.275982734228887,12.418391285961437,28.083458249417557,64.85162422079722,41.60385417427757,13.781436234016974,30.810332395564448,92.5493437956984,14.432187089569673,11.062819234715633,22.73020559967488,12.54289815174759,63.64601572454317,96.64735324771826,40.53561265836049,55.789171181821025,107.12678382765306,35.637879197091074,95.29131347758619,26.049344527142658,65.05604206342134,117.14551683574436,50.36167034101113,94.10603702195378,52.36599647445637,103.49111917292636,118.18940644168174,119.06359958726004,96.57931110151,98.01995530248725,15.04098884234743,21.486276988677165,112.79215860058933,108.75336858673835,18.167363009240034,61.97694226412471,66.91523716843878,109.58181248351065,18.810905185709288,88.99203853827628,97.04196436560474,118.39694647792358,36.68594521254265,92.88890970631307,81.94786248950811,53.91517306876361,50.683753431882494,42.170477576336566,50.159379227391995,46.73362437637008,124.74033986660633,44.75233148155421,112.53435198950012,101.59472610062063,16.754072208599357,104.57985373366743,51.912082089160066,95.93853368418829,98.33770993528772,122.50568581042404,57.309993745800824,92.41014568842833,107.48769524962118,103.30855134302455,132.3314648419572,122.31016285442702,128.48046156620583,100.8489306412824,36.541591223554335,100.52319805532387,16.434410012083415,38.22864275778716,106.87341227757398,51.059849002461135,73.17403503791012,115.48077751284183,132.08054796613882,47.13232673428048,101.37831594925194,86.3959129254102,128.101904128334,86.75078704871397,88.98959979523781,111.98996013334892,113.42025245645607,50.757103918217034,59.55687340753111,55.75169549757503,85.90841176101297,108.37557274179063,14.344647276189244,18.621773967184655,76.37736173606986,107.08139349569015,108.83490789831683,110.8697766427519,53.043743642923374,102.8067661473713,131.98890764461916,104.09562692498336,90.67161208307797,107.61303858924195,9.095587081496378,79.48247113105435,99.69581112705701,9.351198356305456,16.94042707446554,110.94615451610818,111.78046849217202,109.9917048505276,79.22791908746683,83.72341650737488,124.50113361166405,97.76250701183697,93.50755092158435,24.86121751232071,104.23041308090664,118.58365284544506,105.56803864105899,15.291401290251237,16.38789849913935,104.59120794101815,18.49804751104273,105.28990812627372,38.09433070506594,91.94021063051478,92.36773026999015,108.83215700686412,100.92512616394633,99.3597896837326,101.42411905374276,16.662421836932744,16.95246648944354,86.19716395862415,95.27973767616687,94.20767922301705,92.60584184071048,15.243402919912224,96.1188401596983,14.892679675661606,9.906128579583413,35.253577216318035,87.41468258740235,29.390392872966085,39.45641718523938,89.39137688300481,101.03385108010102,12.526136161439094,104.48934492583412,120.34209209298511,15.72662951023576,86.00036743643464,53.640955555623734,102.00309532081054,13.705391992413292,60.38845480791932,104.22640681364916,105.1422297331412,90.87622257194637,80.14500118755376,112.38299812251356,63.92537747901337,105.68376677559402,20.417796462555295,11.696185837375342,40.93135613793305,55.83733197581985,97.4165717886,54.77552034689052,101.00625307570935,122.87682295241596,131.79061898022172,51.290233321617194,88.41845226384976,114.55798204988929,17.10001725377195,86.01163650938116,110.45940758319308,106.8802757080727,102.55690597293899,91.50606528341262,58.27408033583854,128.8235543038411,116.93204258292181,125.34339509074368,15.903292873065935,5.649584532963512]
    a2c_rews = np.array(a2c_rews).reshape(-1,1)
    a2c_x = np.array([i for i in range(len(a2c_rews))]).reshape(-1,1)
    reg = LinearRegression().fit(a2c_x, a2c_rews)
    
    a2c_reg = reg.predict(a2c_x)
    #sns.relplot(
    #    data=a2c_rews, kind="line",
    #    x="iteration", y="A2C_rews", col="region",
    #    hue="event", style="event",
    #)
    tips = sns.load_dataset('tips')
    g = sns.relplot( 
        x="total_bill", y="tip", hue="day",
        col="time", row="sex", data=tips
    )
