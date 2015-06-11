package smodelkit.test;

import static org.junit.Assert.assertEquals;
import static smodelkit.Vector.assertVectorEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import smodelkit.MLSystemsManager;
import smodelkit.Matrix;
import smodelkit.Vector;
import smodelkit.VectorDouble;
import smodelkit.learner.IndependentClassifiers;
import smodelkit.util.Pair;
import smodelkit.util.Range;
import smodelkit.util.ThreadCounter;

public class IndependentClassifiersTest
{

	@Test
	public void innerPredictScoredListOn4ClassSyntheticTest() throws IOException
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'synthetic: -c -3 '\n" + 
				"@ATTRIBUTE x1 NUMERIC\n" + 
				"@ATTRIBUTE x2 NUMERIC\n" + 
				"@ATTRIBUTE x3 NUMERIC\n" + 
				"@ATTRIBUTE class1 {e, f, g, h}\n" + 
				"@ATTRIBUTE class2 {e, f, g, h}\n" + 
				"@ATTRIBUTE class3 {e, f, g, h}\n" + 
				"@DATA\n" + 
				"-0.644794543484926, -1.812068722138941, -1.2959298739425877, f, h, g\n" + 
				"1.038606170341244, 0.6490045899915015, -0.019228398552944434, e, e, f\n" + 
				"0.3231632873327487, -0.576300703310517, 0.5463583480558154, h, h, f\n" + 
				"-0.5129687321070961, -0.7950934383496684, 0.5896731083134469, e, e, e\n" + 
				"0.22740851709284438, 0.5865515881279443, -0.2979798602164796, h, g, f\n" + 
				"0.12583272921950547, -1.209799531345161, 1.7533030969453598, g, g, e\n" + 
				"-0.6176842369555674, 2.2878809918538012, -1.3731875454243216, h, g, f\n" + 
				"-0.2772470258226651, 0.13879462096199024, 0.2897468311181711, g, e, g\n" + 
				"-0.6320839595299511, 2.309204987201548, -1.1248882833684162, g, g, e\n" + 
				"3.098204679254584, 0.0719080980644784, 1.5625184294656027, f, f, f\n" + 
				"-0.4292322627491245, -3.064386379743, 0.03173353723986342, f, e, e\n" + 
				"0.14283291250886626, -1.183157482281388, 1.7296143052063169, g, g, e\n" + 
				"-0.3184202525196172, -1.0508998186627017, 0.5753162821149691, f, e, f\n" + 
				"-0.5765188149611269, -1.8261077582535683, -1.2036483248001673, f, h, h\n" + 
				"-0.43440298890584605, -3.1471151580572605, -0.14324744328424105, f, e, e\n" + 
				"0.4337195250118411, 0.8483920271013711, -1.2932603691520401, g, e, e\n" + 
				"-0.7564595847942612, -1.9656451797366068, -1.3343951941457823, f, h, g\n" + 
				"-1.1496786331577344, 0.2479557076317241, -0.10240204822344058, f, f, f\n" + 
				"-0.5641594682792115, -0.6490088042603704, 1.1375936020828816, h, e, f\n" + 
				"-0.5199700935178503, -3.1147842895724365, 0.003979493035251802, f, e, e\n" + 
				"0.4102099062752593, 0.9584758007176623, -1.1129430375245442, f, e, e\n" + 
				"-0.1660590780856661, 0.3212297972515077, 0.23991472743635808, g, h, f\n" + 
				"0.5271036894459887, -0.8604164402138543, -0.818175749865594, e, g, h\n" + 
				"2.9015163280936602, 0.09008767956282068, 1.5870013357557862, f, f, g\n" + 
				"3.1198924603190465, 0.36118295356554464, 1.4295887666809386, f, f, f\n" + 
				"1.13523041729112, 0.40994346241619894, 0.08279270559252006, f, e, f\n" + 
				"-0.37137457084947345, -3.0003725690225553, 0.23403090133305873, f, e, e\n" + 
				"3.1136623501702, 0.2011114123694783, 1.5401654679506858, f, f, f\n" + 
				"1.2080998791463358, 0.5118774837009661, 0.15779157158437598, e, e, g\n" + 
				"1.164501869610001, 0.5633138285412329, -0.00961187790128483, e, e, f\n" + 
				"0.5153481529415526, 0.9343413889530081, -1.0567070077566063, f, e, e\n" + 
				"-0.8835013674676546, 0.09217608072683339, -0.32831920108751167, g, h, g\n" + 
				"-1.0867255128814004, 0.676564833918456, -0.021803117555288853, g, h, f\n" + 
				"-0.556947954708388, -2.082140146371956, 0.48514043232386866, e, e, g\n" + 
				"0.12282792563229725, -0.5018974422734437, 0.5599013314810474, f, f, f\n" + 
				"0.5057840492612635, -1.2524346592640974, 1.0349045308638685, f, f, f\n" + 
				"-0.8139699844405186, -0.488845075798999, 1.1457897097117216, h, h, h\n" + 
				"0.02844847388674332, 0.3903869989538414, 0.15269055903941103, g, h, g\n" + 
				"0.29037273054724416, 0.9578798576152442, -1.008112401498804, f, e, e\n" + 
				"0.3668427154793058, -0.8113500747060903, -0.5508247547949886, h, e, e\n" + 
				"-1.1154582746108654, 0.4249935858287357, -0.08089561830046316, g, h, g\n" + 
				"-0.45074635822007864, -3.1568185861650764, -0.12371035892432672, f, e, e\n" + 
				"1.1804914974399112, 1.3268424836032664, -1.5426819344061746, f, e, e\n" + 
				"-0.6547143143978583, -1.7161320928993755, -1.2530331266167094, f, h, h\n" + 
				"-0.7031065564041623, -2.03055252670576, 0.5811631490742097, h, h, h\n" + 
				"0.3938418473266231, 0.9454334893305636, -1.2229871748283037, f, e, e\n" + 
				"2.7417306317227883, 0.34615834897103026, 1.5698005324445958, h, e, e\n" + 
				"-1.433144013507755, -0.5073956345442544, -1.3152106686846434, f, e, e\n" + 
				"0.26120790939308713, 0.4698869986570764, -0.09595227443194282, f, e, f\n" + 
				"-0.4253198407367845, -0.9942751845084207, 0.5165737349728274, f, e, f\n" + 
				"-0.025688624759932888, 0.34762756018606006, 0.22441575533246869, g, h, g\n" + 
				"0.13304604810651355, -0.8999291165608668, 1.641198132439045, g, g, e\n" + 
				"0.0859929592979097, -0.0814862489249393, -0.09663446614305345, g, g, e\n" + 
				"3.0590806658914107, 0.3545348845433447, 1.734752271097942, h, e, f\n" + 
				"0.3489671902030448, 0.5889004417385747, -0.12235969738403314, f, f, f\n" + 
				"-0.8837039568808551, -0.5609804190436986, 1.210635164689976, h, h, h\n" + 
				"0.15071152205613172, -1.2318454635540501, 1.5360977900840815, g, g, e\n" + 
				"-0.0015597133485246578, -1.2701820357852243, 1.5985250093340166, e, e, f\n" + 
				"0.9527965340047406, 1.2635023161200574, -1.3679264428119768, e, e, e\n" + 
				"0.27838758576148254, -1.0613727064565812, 1.6700912208665113, g, g, e\n" + 
				"0.37741981326685503, 0.8476015900740274, -1.23744440789548, f, e, e\n" + 
				"-0.638427283680928, 2.3893939532585597, -1.285878732195319, h, g, f\n" + 
				"-0.6900063248147872, -0.5516743446380186, 1.251392510049519, g, h, g\n" + 
				"0.37238740218817046, 0.45296801632834516, -0.29092743228624535, f, f, f\n" + 
				"0.29665324636628104, -1.3752730842767527, 1.6150806049562498, f, g, e\n" + 
				"0.5285804078614688, -0.859159985937995, -0.721945628991467, e, g, h\n" + 
				"0.06732830503143629, -0.4959794555296572, 0.7911959736461482, h, g, f\n" + 
				"-0.3539880333760791, -1.0695473128835444, 0.5672893048212966, f, f, f\n" + 
				"0.3146737963024337, -1.2772546679373364, 1.6580436671494763, g, g, e\n" + 
				"-0.08842500008305032, 0.2064736460271584, 0.25742229743782125, g, h, g\n" + 
				"0.6642057250875301, -0.8572063335244167, -0.7352743173766333, e, g, h\n" + 
				"-0.4699415873445137, -0.869514531013354, 0.39706711534643724, e, e, e\n" + 
				"0.9796551052688797, 1.2570550388454815, -1.6296662000951168, e, e, h\n" + 
				"-1.4537261002326116, -0.49506063188748145, -1.1613629551258933, e, e, e\n" + 
				"0.6194971325110703, -0.828341347209636, -0.5583096801383125, h, g, h\n" + 
				"-0.15976993018347763, -0.7626732861625666, 0.5128469043034268, e, e, e\n" + 
				"0.4944731671800914, -0.8284188758230108, -0.9065283615122905, h, e, e\n" + 
				"-0.47408419569566657, -0.4631210316165078, 0.976545622805904, e, e, f\n" + 
				"0.055005996165530285, -0.5542628643549381, 0.5849922017278059, h, g, f\n" + 
				"-0.4686119777096116, 2.4222451960380447, -1.3566396829416274, g, g, e\n" + 
				"3.136796187573107, 0.2748982629158397, 1.5838489616814124, f, f, f\n" + 
				"0.09999086145276258, -1.258753283734409, 1.8472718418204386, e, e, f\n" + 
				"0.3199765357498888, 0.04080594817640349, -0.09579063878185916, f, f, f\n" + 
				"-0.6198881098461868, -0.46463256845585743, 0.927904416299689, e, f, f\n" + 
				"0.16514154844191062, -0.07519764420920874, -0.031032818635179357, f, f, f\n" + 
				"-1.174581355093448, 0.37131478882200886, -0.08153860657276552, f, f, f\n" + 
				"-0.9874842352952121, 0.40037962274206956, -0.13369353728457856, g, e, g\n" + 
				"-0.1428723106317294, 0.07656025408538036, 0.23669471180395335, g, h, g\n" + 
				"-1.015429822139395, 0.2337597314940542, -0.07867588040578516, g, h, g\n" + 
				"-0.5567505725441311, -3.1619111185971045, 0.009758175643167805, f, e, e\n" + 
				"0.6702713178327271, -0.9500898950105039, -0.8546864353220719, h, e, e\n" + 
				"0.9314811854125162, 1.3685664979465675, -1.5414929170434568, e, e, e\n" + 
				"-0.9876752045928421, -0.39983534139256505, 1.149489267772946, h, h, h\n" + 
				"0.7709229024550591, -1.2280642869076548, 1.1425502734259494, h, h, h\n" + 
				"-0.6070256570028413, -1.5508978154378363, -1.3668975902458214, f, e, g\n" + 
				"-0.4849885292147382, 2.302680173971003, -1.297441534182188, h, g, f\n" + 
				"0.244404385766759, 0.3899033030004351, -0.23621906588685007, e, g, f\n" + 
				"0.4002329838482909, -0.8176558787879652, -0.7584065906030942, e, e, e\n" + 
				"-0.526536129625116, -3.164943794472278, 0.2108309951200685, f, e, e\n" + 
				"0.32619426495396997, -0.11694731274460415, -0.03388686576859941, e, e, e\n" + 
				"0.5293737006831396, -0.865089582443064, -0.7990155388051252, h, g, h\n" + 
				"0.5636146841844939, 0.8508671849968885, -1.2332459782853917, f, e, e\n" + 
				"-0.9566438887373494, -0.5224978675479182, 1.274747727878472, h, h, h\n" + 
				"-1.0968000941146625, 0.36315926999962883, -0.1621148030772463, g, h, h\n" + 
				"-0.32628214679539547, -0.8565596493064157, 0.5621389021518526, e, e, f\n" + 
				"-0.4535860163557782, -1.0192943081734975, 0.7314156198921373, e, e, e\n" + 
				"-0.592685393299265, -1.8062741498314452, 0.5171216484930676, e, h, h\n" + 
				"-1.5635223336400241, -0.34340546339665856, -1.344000934791231, e, e, e\n" + 
				"-1.6455342166168858, -0.26211843796033607, -1.305558241772278, e, h, e\n" + 
				"0.5176223194830376, 0.38771646103699386, -0.27376776178928697, h, g, g\n" + 
				"-0.6394168098741299, 2.3106715820478625, -1.4155742499175852, h, g, f\n" + 
				"-0.7179435398373412, 2.27548452306469, -1.4027855114830416, h, g, f\n" + 
				"0.15397754510348188, -0.018580521751495285, -0.1686447571353474, f, g, f\n" + 
				"-0.6153619247357129, -3.1787959357806423, -0.10658789620612374, f, e, e\n" + 
				"0.14804682077161807, -1.1084769047641236, 1.7537522880324876, e, e, f\n" + 
				"3.13768776934299, 0.15085949205741006, 1.5108607191206205, e, e, e\n" + 
				"-0.7446892087951331, -1.6672636365316507, -1.176738762071537, g, h, g\n" + 
				"0.5057025662433795, -0.9523504850738321, -0.8386112399420759, e, g, h\n" + 
				"0.17275201680151867, 0.12074782796496553, -0.310550516082445, h, g, e\n" + 
				"0.6157459768313912, -0.8777414620822064, -0.9512249433394209, e, g, h\n" + 
				"-0.6258710013481319, 2.2782567793098956, -1.2426466196921282, h, g, f\n" + 
				"0.5074799300082626, -0.9363168898567095, -0.6793543676249583, e, g, h\n" + 
				"0.5519989365490222, -0.7600170404995344, -0.657085619162488, h, f, e\n" + 
				"0.6041126827350538, -0.7528227965757672, -0.6664050976761531, e, g, h\n" + 
				"0.3736596204300159, -0.04707071946529308, -0.0563598986255074, h, h, e\n" + 
				"-1.4871528436439934, -0.2709096282915561, -1.0582790585722954, e, e, e\n" + 
				"-0.4566470073693223, 2.306635913815796, -1.3596117824108893, g, h, e\n" + 
				"-1.0987526474872167, 0.2208521453845419, -0.04879799768374936, f, f, f\n" + 
				"0.18165779018882552, 0.004487405461414934, -0.14614698155180955, h, g, h\n" + 
				"2.9438547254506395, 0.34529382750710785, 1.4008936253057411, f, f, f\n" + 
				"-0.4176068073692103, -1.2090689781008972, 0.6340871021161115, e, e, e\n" + 
				"0.04004688552032855, -1.2027088887237196, 1.7402539123408263, e, e, e\n" + 
				"0.07950036944590846, -1.4260003126686702, 1.65314571391027, e, e, f\n" + 
				"0.5673161310003456, -0.9062977567575274, -0.8052328045981588, h, e, e\n" + 
				"-0.40767958076442495, -2.9496747389319773, -0.02492183627363343, f, e, e\n" + 
				"-1.657623931368562, -0.5191823497878958, -1.318924992972286, e, e, e\n" + 
				"2.938136630888236, 0.16921916412230054, 1.4607203301028182, f, f, f\n" + 
				"0.12536674220431393, -0.206884659670328, -0.2774991501145858, e, g, e\n" + 
				"-0.4025991868661941, -3.068939705092946, 0.06262445174796305, f, e, e\n" + 
				"0.09021271188537772, -1.1909536813208352, 1.7419390339606586, f, f, e\n" + 
				"0.3005203763092933, 0.4732695016770919, -0.12764503764514434, h, g, f\n" + 
				"-0.30341444425184894, -0.9459775231793561, 0.5595615948864305, f, e, f\n" + 
				"-0.6716871322087982, 2.3384696769900204, -1.2593518850004666, g, g, e\n" + 
				"0.2800265853167302, -0.013171866929393797, 0.10964989976034406, h, g, e\n" + 
				"0.5343089974965473, 0.85348153628787, -1.0697954853886313, f, e, e\n" + 
				"-0.818672179987052, -0.6194648304797489, 1.1770208627369716, h, h, h\n" + 
				"-1.5546156154644621, -0.4625286897315768, -1.162440521969551, e, h, e\n" + 
				"0.9383366643903703, 1.2157156241835652, -1.3449861302679669, e, e, e\n" + 
				"-0.44076210066328925, -3.039297889319247, -0.09393519230909195, f, e, e\n" + 
				"0.43405476942850196, 0.7733042661009885, -1.2364194292835817, f, e, e\n" + 
				"2.872934194986595, 0.303429505762262, 1.633461799121768, h, h, f\n" + 
				"-0.7192217815746809, -1.8612730432168136, 0.5832682263147879, g, h, h\n" + 
				"0.9820883955000447, 1.4185821011887725, -1.4469785674346627, e, e, e\n" + 
				"-1.1866791566655948, 0.22103533787455054, -0.09070782963620028, f, f, f\n" + 
				"0.19563338916279605, 0.04818091471202714, -0.29542674265750013, f, f, f\n" + 
				"0.4668363941903999, -0.8207567899309978, -0.6893997938501604, e, g, h\n" + 
				"0.993572280843007, 0.5215090619747725, 0.11084604794184963, e, e, f\n" + 
				"-0.5045829455497265, -0.6062739939679866, 0.9567811017970494, h, g, f\n" + 
				"-0.48670274488939397, 2.172534945618871, -1.397556953504621, h, g, f\n" + 
				"1.0272099100456709, 1.3164040922805198, -1.3514879476651172, h, e, e\n" + 
				"-0.4154778361648354, -0.9006513107654834, 0.37855066876779325, f, e, f\n" + 
				"-0.3488305821993415, -0.5425654813775344, 1.108053903454334, f, f, f\n" + 
				"0.7523579065465036, -0.9252987647618615, -0.821415407465362, h, e, e\n" + 
				"1.156225998182147, 0.42059936284499366, 0.16742334152033067, e, e, f\n" + 
				"-0.5215442522495025, -3.2352020382539304, -0.12663039748854316, f, e, e\n" + 
				"-0.7047039741688085, -2.0278538524320853, -1.294842418424211, f, h, g\n" + 
				"1.3068459911000663, 0.6549936757160639, 0.06103628463869954, e, e, f\n" + 
				"0.5301728863073472, -0.8036533548744623, -0.6631222081078444, h, h, f\n" + 
				"0.48472694434629715, -1.2621356747333476, 1.0432729176231978, g, h, h\n" + 
				"-0.31161837515991825, -0.9871585984536458, 0.508125597767019, f, f, f\n" + 
				"0.6166076955335802, -0.6924092880819928, -0.7346605066670275, e, h, h\n" + 
				"1.0902165000169854, 1.1912158836936004, -1.6908168917306734, e, e, e\n" + 
				"0.19002048630863116, -1.1326628692744583, 1.796769698137383, e, e, f\n" + 
				"0.5700534490392293, -1.1727596493715302, 1.158985517047627, f, f, f\n" + 
				"-1.4211882005640724, -0.45632667197870724, -1.1904149772075088, e, f, e\n" + 
				"0.3415892783251159, 0.4786239674512589, -0.15138482121199523, f, f, f\n" + 
				"-1.602697898722143, -0.4739229375991119, -1.3253649607413536, e, e, e\n" + 
				"0.6042305045078207, -0.8155383006648504, -0.6786386713045083, e, g, h\n" + 
				"0.3434104590066506, -1.2085428525455588, 1.6711195566060684, g, g, e\n" + 
				"0.016497261350082115, -0.382320109336304, 0.6545562531306056, h, g, f\n" + 
				"-0.22243454579913807, 0.1873375709540963, 0.100708672352948, g, h, g\n" + 
				"0.49347314106013696, 0.8953070074851253, -1.2599728282622988, e, e, e\n" + 
				"3.0498031496242364, 0.2413547084495152, 1.580426834434879, g, h, g\n" + 
				"-0.7260468661263852, -1.8458001561315118, 0.6856416176221594, e, e, h\n" + 
				"0.5361081661947578, 0.9904195479129664, -0.9893838541371162, f, e, e\n" + 
				"-0.2389623829600591, 0.1596292628996957, 0.32124147096101285, g, g, g\n" + 
				"0.5852736813916632, -0.8249939812433703, -0.7407768933001639, e, g, h\n" + 
				"-0.5454378436188855, -1.7312874678747572, -1.3458588688610102, f, e, g\n" + 
				"1.1377131983174424, 1.1531110313427804, -1.4666842103157933, f, e, e\n" + 
				"-0.6649840365514955, -1.771235879212244, -1.206794969679878, f, h, h\n" + 
				"-0.564701983968538, -1.653763215821281, -1.4852282618423542, f, h, g\n" + 
				"0.0692897007652371, -1.194675216793675, 1.8345666553541309, g, g, f\n" + 
				"1.034409144776116, 0.6237471938329684, 0.03172104467960517, e, e, f\n" + 
				"0.33860302364716865, 0.4508798622603705, -0.327749953172487, f, f, f\n" + 
				"0.2842106428359069, 0.3844258089914684, -0.057751585889203816, h, h, f\n" + 
				"-1.7668389127325872, -0.5957075064125676, -1.1984905132349322, e, e, e\n" + 
				"2.9729620580873113, 0.1548296661629112, 1.5560934013981917, f, f, f\n" + 
				"-0.24886508995375473, -0.43974603453084055, 0.9629126432034294, e, e, f\n" + 
				"0.9515552501678943, 1.2305795609216534, -1.5550275357250656, e, e, e\n" + 
				"0.9848363984141026, 0.4521800317421367, 0.1848689647949383, h, h, f\n" + 
				"-0.4585954076973425, -1.7570567456505053, -1.2583697905902425, f, h, g\n" + 
				"0.5654681084239538, -1.3034124102019107, 1.038546981687663, f, f, f\n" + 
				"-0.6413183814440462, -3.201724817696811, 0.01153290302383693, f, e, e\n" + 
				"0.16025345791999748, 0.448692228665218, -0.14772261752992893, h, g, f\n" + 
				"-0.4950595804188363, -1.811598843498008, -1.2855866976818846, f, h, g\n" + 
				"-0.5222093630202841, -0.6125857003125184, 0.8791751425274301, e, e, f\n" + 
				"0.43399734708935334, -0.7502957325642067, -0.639362475427316, g, h, g\n" + 
				"-0.7108048853336786, -0.6219164069596033, 1.1510408232793692, h, h, e\n" + 
				"-0.18706118269672473, 0.2813073431391403, 0.24519798440801438, g, h, g\n" + 
				"-0.6007936963896775, -2.0120799332807895, 0.4878587153487787, e, h, g\n" + 
				"2.9723818001447833, 0.21921275875845417, 1.4387570916566832, f, f, f\n" + 
				"-0.749532760791958, -0.691891884887819, 1.1792478274756408, g, g, h\n" + 
				"1.1288085748382546, 0.5745833409191489, 0.037594256609050095, e, e, f\n" + 
				"0.10009838957451618, 0.043944082070837415, -0.12696373409563064, f, f, f\n" + 
				"-0.7757694576089627, -1.9601679715219658, 0.41310172788839783, e, e, g\n" + 
				"-0.03799425372313994, 0.10631879363974026, 0.3107338989799077, g, e, g\n" + 
				"0.3409085971257666, 0.0333126883843411, -0.09296521509723724, h, g, e\n" + 
				"0.2162114059752251, -0.03945810160346958, -0.2633405355053785, f, f, f\n" + 
				"-0.5287486174557006, -1.9154321788096127, -1.24890034458486, e, h, g\n" + 
				"0.9071052366950103, 1.3733915373817265, -1.3281725674680518, h, h, h\n" + 
				"1.088517477335324, 0.4349601893112849, 0.10557646229465156, e, e, f\n" + 
				"0.9973864695043493, 0.4693332333138414, 0.05862753074546368, e, e, f\n" + 
				"0.3825966504346301, 1.035057311098162, -1.055599550820373, f, e, e\n" + 
				"-0.4004155513750035, -0.5031279751240436, 0.9534894378999506, e, e, f\n" + 
				"-0.6962188342803831, 2.3285791334697823, -1.253621523949001, g, g, e\n" + 
				"1.0072369201673657, 0.6114032340361221, -0.007563565997143237, e, e, f\n" + 
				"0.42880337779187017, 0.9071490405505028, -1.2379539088608433, f, e, e\n" + 
				"1.1397620891386464, 1.206560458195848, -1.3039386461139397, e, h, e\n" + 
				"-0.18784601006845075, 0.055822420021407415, 0.23676555409479977, g, h, g\n" + 
				"-0.6537960932131385, -3.1934893229444117, -0.0019767387301124206, f, f, f\n" + 
				"2.858291082835524, 0.16549935234455976, 1.5548547159602575, g, h, g\n" + 
				"0.4721584003680548, 0.3518756251894331, -0.44287453455956793, g, g, f\n" + 
				"-1.1160185206107398, 0.19224846840505333, 0.030628459426503052, f, f, f\n" + 
				"-1.4911863676445316, -0.3780398442460171, -1.1095159657764244, e, e, e\n" + 
				"-0.5324785276822882, -0.4759769841297157, 1.1255413864863548, e, e, f\n" + 
				"-0.45817249454908693, -3.2253809575978374, -0.05534587490424526, f, e, e\n" + 
				"0.4637622585542633, -1.035313790978535, -0.7956052527700619, g, f, e\n" + 
				"-1.0976774396079136, 0.4950406007315158, -0.13589033174847445, g, h, f\n" + 
				"0.38354352025572414, 0.46303286647573255, -0.13584577904698245, h, g, f\n" + 
				"0.5997565887758962, -0.910520547358311, -0.6719160347016999, e, f, h\n" + 
				"-1.7261060744919237, -0.4646355680744317, -1.1628973450272087, e, e, e\n" + 
				"-1.0579009258398653, 0.10356812234594179, -0.10805761614839396, g, h, g\n" + 
				"-0.7080767953709338, -1.9571147908163182, 0.5514248274907608, e, e, g\n" + 
				"3.015979814064375, 0.22355751063520846, 1.6482145518469171, f, f, f\n" + 
				"-0.2249942157781596, -1.1363614267486062, 0.3964775048180833, e, e, e\n" + 
				"-0.3151714879095503, -1.0900344888155016, 0.3510487217547198, f, f, f\n" + 
				"-0.4373394899274272, -0.505199628277228, 1.1045961081201685, e, e, f\n" + 
				"1.0450806071748375, 0.662003236413878, 0.030261346979937903, e, e, f\n" + 
				"-0.7499905201222469, 2.4090279790094904, -1.272667801032282, g, g, e\n" + 
				"-0.7845758888884035, -1.8551983144782362, 0.5189391381465565, e, h, h\n" + 
				"-0.4546877896032219, -0.6356703084630313, 0.8966153329753561, h, g, e\n" + 
				"0.671694347850564, -0.8736634902816676, -0.8594888407343948, h, e, e\n" + 
				"0.5555171297257857, 1.00864006289705, -1.3382176029524646, f, e, e\n" + 
				"-0.40353362152837013, -0.4537643310091126, 0.9043590448006903, e, e, f\n" + 
				"-3.913841554898101E-4, -0.3933280986160363, 0.7235136432795368, f, f, f\n" + 
				"-0.6179395188558408, -1.9806857858548865, -1.396517255581235, g, g, g\n" + 
				"0.3400284518236314, -1.0259341750686246, -0.5258750316944761, e, g, e\n" + 
				"2.88688534678478, 0.23820139294701134, 1.643699585512502, f, f, f\n" + 
				"0.0713403714731666, 0.020324530163966313, -0.045424292358897556, h, g, e\n" + 
				"3.2661799824790547, 0.2375493455427877, 1.515178513862351, h, e, e\n" + 
				"-0.5717487048015768, -1.9289262478469773, -1.2535337427447515, f, h, g\n" + 
				"-0.8006545902667805, -0.5071857367088679, 1.2514192197195602, g, h, h\n" + 
				"-0.5582402080075457, -0.8679867927191984, 0.5445048580516171, f, e, e\n" + 
				"-0.57214765187999, -0.7037176134679466, 1.0204763718127994, e, e, f\n" + 
				"-0.06377996531119713, -1.1778693314132722, 1.7760767404605828, e, e, f\n" + 
				"0.2577407173108033, 0.4944271161386763, -0.13613743652186505, h, g, f\n" + 
				"-0.026944221523293482, -0.4743052927249461, 0.7857191883406744, f, f, f\n" + 
				"-0.0037967894976464744, 0.12555447746478124, 0.4816356950680272, f, h, f\n" + 
				"0.47373276459905567, -1.0959557395143007, 1.1912097233841874, h, h, h\n" + 
				"2.9509661505864324, 0.36081829722121866, 1.583963113910448, f, f, e\n" + 
				"2.9791901763370268, 0.07352447607001072, 1.441110730550803, f, f, f\n" + 
				"0.7135799037474926, -1.1075067346195242, 1.1323572051394795, h, h, h\n" + 
				"-0.5361578015006404, -3.261837098066393, 0.09178553990024235, f, h, e\n" + 
				"-0.605024637111071, -3.1540690569305165, -0.15154574514282657, f, e, e\n" + 
				"0.5651591933347474, -1.044327121759344, 0.993774451747596, e, f, f\n" + 
				"-0.18055984131426397, 0.13169427867003786, 0.2854912122820774, g, h, g\n" + 
				"-1.0592995505306555, 0.25387776328259176, -0.15511228571084748, f, f, f\n" + 
				"-0.06970469847667558, 0.1023490937807073, 0.27791240761243613, h, h, g\n" + 
				"-0.33316621904854815, -0.5630764239411858, 1.1500003806925563, f, g, f\n" + 
				"2.8480661388201267, 0.2519354613186047, 1.4815811008137918, g, h, g\n" + 
				"0.35586596220120786, 0.8207094567042692, -1.0817326624440187, f, e, e\n" + 
				"-0.6740614278635348, -0.5608549795264157, 1.0844328563798757, h, h, h\n" + 
				"0.6534276771280085, -1.2770677817541762, 1.047954134665668, h, h, h\n" + 
				"-1.6902705399865867, -0.14331041013318774, -1.2932268129246711, e, e, e\n" + 
				"1.0197446344667631, 1.312598233091505, -1.4941979350539063, e, e, e\n" + 
				"-0.8154375733861621, -0.5957906535236596, 1.2034226061957836, h, h, h\n" + 
				"-0.6933176613828758, -1.9395267880792162, 0.7037594475742805, e, e, f\n" + 
				"-1.1260769308210699, 0.39434593558406067, -0.27444771473325635, g, h, g\n" + 
				"0.5596121699985043, 0.8893326728483854, -1.1173009651626877, f, e, e\n" + 
				"0.4719878868296094, -1.141349723209549, 1.1589663599699942, h, h, h\n" + 
				"0.5229266207183565, -0.7758083522531065, -0.7165736696055565, e, g, h\n" + 
				"-1.790939794730596, -0.4599701580092331, -1.1603063822658646, e, e, e\n" + 
				"0.23017485646464841, -0.1040136447847322, -0.1359936909749097, f, g, f\n" + 
				"-0.5470420599715577, -2.191436374003468, 0.4497870884827119, h, h, f\n" + 
				"-0.956962713893609, -0.6196743166368296, 1.281798996239326, h, h, h\n" + 
				"2.8209502064746497, 0.08086358236530278, 1.5770955332700045, h, e, e\n" + 
				"");
		
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		ThreadCounter.setMaxThreads(1);
		IndependentClassifiers ic = (IndependentClassifiers)MLSystemsManager.createLearner(new Random(0), "ic", "model_settings/ic.json");
		ic.train(inputs, labels);
		
		// These were generated using my old code that exhaustively explored every output vector.		
		List<Vector> expected = Arrays.asList(
			Vector.create(new double[]{1.0, 0.0, 2.0}, 0.15284057122543168),
			Vector.create(new double[]{1.0, 3.0, 2.0}, 0.09969567840333066),
			Vector.create(new double[]{1.0, 2.0, 2.0}, 0.08976553529322807),
			Vector.create(new double[]{0.0, 0.0, 2.0}, 0.07990163488210617),
			Vector.create(new double[]{1.0, 1.0, 2.0}, 0.06801126923526359),
			Vector.create(new double[]{0.0, 3.0, 2.0}, 0.052118672622320956),
			Vector.create(new double[]{1.0, 0.0, 3.0}, 0.04846546350953347),
			Vector.create(new double[]{0.0, 2.0, 2.0}, 0.04692741572797053),
			Vector.create(new double[]{3.0, 0.0, 2.0}, 0.04640255980925351),
			Vector.create(new double[]{0.0, 1.0, 2.0}, 0.03555477160766089),
			Vector.create(new double[]{1.0, 3.0, 3.0}, 0.03161331592112518),
			Vector.create(new double[]{3.0, 3.0, 2.0}, 0.030267713884760038),
			Vector.create(new double[]{1.0, 2.0, 3.0}, 0.028464485838324217),
			Vector.create(new double[]{3.0, 2.0, 2.0}, 0.027252911886268744),
			Vector.create(new double[]{2.0, 0.0, 2.0}, 0.02622760463410918),
			Vector.create(new double[]{0.0, 0.0, 3.0}, 0.02533666119331036),
			Vector.create(new double[]{1.0, 0.0, 0.0}, 0.023710320705714997),
			Vector.create(new double[]{1.0, 1.0, 3.0}, 0.02156624815604103),
			Vector.create(new double[]{3.0, 1.0, 2.0}, 0.02064829359828669),
			Vector.create(new double[]{2.0, 3.0, 2.0}, 0.017107884483336524),
			Vector.create(new double[]{0.0, 3.0, 3.0}, 0.0165267350539849),
			Vector.create(new double[]{1.0, 3.0, 0.0}, 0.015465896842470519),
			Vector.create(new double[]{2.0, 2.0, 2.0}, 0.015403861360655613),
			Vector.create(new double[]{0.0, 2.0, 3.0}, 0.014880597058264751),
			Vector.create(new double[]{3.0, 0.0, 3.0}, 0.014714166213546017),
			Vector.create(new double[]{1.0, 2.0, 0.0}, 0.013925423158641459),
			Vector.create(new double[]{0.0, 0.0, 0.0}, 0.012395225775304157),
			Vector.create(new double[]{2.0, 1.0, 2.0}, 0.011670806160070397),
			Vector.create(new double[]{0.0, 1.0, 3.0}, 0.011274352563098508),
			Vector.create(new double[]{1.0, 1.0, 0.0}, 0.01055066068022197),
			Vector.create(new double[]{3.0, 3.0, 3.0}, 0.009597836301168889),
			Vector.create(new double[]{3.0, 2.0, 3.0}, 0.008641848142561194),
			Vector.create(new double[]{2.0, 0.0, 3.0}, 0.008316725102146053),
			Vector.create(new double[]{0.0, 3.0, 0.0}, 0.008085225229947172),
			Vector.create(new double[]{0.0, 2.0, 0.0}, 0.007279900015287663),
			Vector.create(new double[]{3.0, 0.0, 0.0}, 0.007198478557246144),
			Vector.create(new double[]{3.0, 1.0, 3.0}, 0.006547535853198789),
			Vector.create(new double[]{0.0, 1.0, 0.0}, 0.005515649612383924),
			Vector.create(new double[]{2.0, 3.0, 3.0}, 0.005424878646452589),
			Vector.create(new double[]{2.0, 2.0, 3.0}, 0.004884536054105935),
			Vector.create(new double[]{3.0, 3.0, 0.0}, 0.004695462713090605),
			Vector.create(new double[]{3.0, 2.0, 0.0}, 0.004227773266006381),
			Vector.create(new double[]{2.0, 0.0, 0.0}, 0.004068716259246427),
			Vector.create(new double[]{2.0, 1.0, 3.0}, 0.003700791128577076),
			Vector.create(new double[]{3.0, 1.0, 0.0}, 0.0032031917920474036),
			Vector.create(new double[]{2.0, 3.0, 0.0}, 0.0026539643528153706),
			Vector.create(new double[]{2.0, 2.0, 0.0}, 0.002389617429712563),
			Vector.create(new double[]{2.0, 1.0, 0.0}, 0.0018105045979012874),
			Vector.create(new double[]{1.0, 0.0, 1.0}, 7.371651808405386E-4),
			Vector.create(new double[]{1.0, 3.0, 1.0}, 4.8084211024580884E-4),
			Vector.create(new double[]{1.0, 2.0, 1.0}, 4.32948048591627E-4),
			Vector.create(new double[]{0.0, 0.0, 1.0}, 3.8537348202164916E-4),
			Vector.create(new double[]{1.0, 1.0, 1.0}, 3.280250733364533E-4),
			Vector.create(new double[]{0.0, 3.0, 1.0}, 2.5137350914590935E-4),
			Vector.create(new double[]{0.0, 2.0, 1.0}, 2.2633556407261344E-4),
			Vector.create(new double[]{3.0, 0.0, 1.0}, 2.2380413210311632E-4),
			Vector.create(new double[]{0.0, 1.0, 1.0}, 1.7148417747829157E-4),
			Vector.create(new double[]{3.0, 3.0, 1.0}, 1.4598417554053338E-4),
			Vector.create(new double[]{3.0, 2.0, 1.0}, 1.3144348753735705E-4),
			Vector.create(new double[]{2.0, 0.0, 1.0}, 1.2649832932514043E-4),
			Vector.create(new double[]{3.0, 1.0, 1.0}, 9.958876077464458E-5),
			Vector.create(new double[]{2.0, 3.0, 1.0}, 8.251301770101844E-5),
			Vector.create(new double[]{2.0, 2.0, 1.0}, 7.429434576518286E-5),
			Vector.create(new double[]{2.0, 1.0, 1.0}, 5.6289451580586804E-5));

		
		List<Vector> actual = ic.predictScoredList(inputs.row(0), 1000);
		
		// This prints out the above expected results.
//		for (Tuple2<double[], Double> tuple : actual)
//		{
//			System.out.println("Vector.create(new double[]" 
//					+ Arrays.toString(tuple.getFirst()).replace("[", "{").replace("]", "}") + ", " + tuple.getSecond() + "),");
//		}
		
		assertEquals(expected.size(), actual.size());
		for (int i : new Range(expected.size()))
		{
			assertVectorEquals(expected.get(i), actual.get(i), 0.0);
		}
		
	}
	
	@Test
	public void innerPredictScoredListOn2ClassSyntheticTest() throws IOException
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'synthetic: -c -2 '\n" + 
				"@ATTRIBUTE x1 NUMERIC\n" + 
				"@ATTRIBUTE x2 NUMERIC\n" + 
				"@ATTRIBUTE x3 NUMERIC\n" + 
				"@ATTRIBUTE class1 {e, f}\n" + 
				"@ATTRIBUTE class2 {e, f}\n" + 
				"@DATA\n" + 
				"-0.644794543484926, -1.812068722138941, -1.2959298739425877, f, f\n" + 
				"1.038606170341244, 0.6490045899915015, -0.019228398552944434, e, e\n" + 
				"-0.6179395188558408, -1.9806857858548865, -1.396517255581235, f, f\n" + 
				"0.3400284518236314, -1.0259341750686246, -0.5258750316944761, e, f\n" + 
				"2.88688534678478, 0.23820139294701134, 1.643699585512502, f, f\n" + 
				"0.0713403714731666, 0.020324530163966313, -0.045424292358897556, e, f\n" + 
				"3.2661799824790547, 0.2375493455427877, 1.515178513862351, e, e\n" + 
				"-0.5717487048015768, -1.9289262478469773, -1.2535337427447515, f, f\n" + 
				"-0.8006545902667805, -0.5071857367088679, 1.2514192197195602, f, f\n" + 
				"-0.5582402080075457, -0.8679867927191984, 0.5445048580516171, f, e\n" + 
				"-0.57214765187999, -0.7037176134679466, 1.0204763718127994, e, e\n" + 
				"-0.06377996531119713, -1.1778693314132722, 1.7760767404605828, e, e\n" + 
					"");
		
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		ThreadCounter.setMaxThreads(1);
		IndependentClassifiers ic = (IndependentClassifiers)MLSystemsManager.createLearner(new Random(0), "ic", "model_settings/ic.json");
		ic.train(inputs, labels);
		
		// These were generated using my old code that exhaustively explored every output vector.
		List<Vector> expected = Arrays.asList(
				Vector.create(new double[]{1.0, 1.0}, 0.28534662063805905),
				Vector.create(new double[]{0.0, 1.0}, 0.2793039608983343),
				Vector.create(new double[]{1.0, 0.0}, 0.2383080748259086),
				Vector.create(new double[]{0.0, 0.0}, 0.23326152965855448)
				);
		
		List<Vector> actual = ic.predictScoredList(inputs.row(0), 100);
		
		// This prints out the above expected results.
//		for (Tuple2<double[], Double> tuple : actual)
//		{
//			System.out.println("Vector.create( new double[]" 
//					+ Arrays.toString(tuple.getFirst()).replace("[", "{").replace("]", "}") + ", " + tuple.getSecond() + "),");
//		}
		
		assertEquals(expected.size(), actual.size());
		for (int i : new Range(expected.size()))
		{
			assertVectorEquals(expected.get(i), actual.get(i), 0.0);
		}
		
	}


}
