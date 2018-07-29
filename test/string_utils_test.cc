#include <primitiv/config.h>

#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/string_utils.h>

using std::string;
using std::vector;

namespace primitiv {
namespace string_utils {

class StringUtilsTest : public testing::Test {};

TEST_F(StringUtilsTest, CheckJoin) {
  EXPECT_EQ("", join(vector<string> {}, ""));
  EXPECT_EQ("", join(vector<string> {}, "."));
  EXPECT_EQ("", join(vector<string> {}, "xxx"));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, ""));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, "."));
  EXPECT_EQ("foo", join(vector<string> {"foo"}, "xxx"));
  EXPECT_EQ("foobar", join(vector<string> {"foo", "bar"}, ""));
  EXPECT_EQ("foo.bar", join(vector<string> {"foo", "bar"}, "."));
  EXPECT_EQ("fooxxxbar", join(vector<string> {"foo", "bar"}, "xxx"));
  EXPECT_EQ("foobarbaz", join(vector<string> {"foo", "bar", "baz"}, ""));
  EXPECT_EQ("foo.bar.baz", join(vector<string> {"foo", "bar", "baz"}, "."));
  EXPECT_EQ("fooxxxbarxxxbaz", join(vector<string> {"foo", "bar", "baz"}, "xxx"));
}

TEST_F(StringUtilsTest, CheckToString) {
  // Generated by std::to_string()
  EXPECT_EQ("0", to_string(0));
  EXPECT_EQ("5", to_string(5));
  EXPECT_EQ("123", to_string(123));
  EXPECT_EQ("2147483647",
            to_string(std::numeric_limits<std::int32_t>::max()));
  EXPECT_EQ("-2147483648",
            to_string(std::numeric_limits<std::int32_t>::min()));
  EXPECT_EQ("0", to_string(0u));
  EXPECT_EQ("5", to_string(5u));
  EXPECT_EQ("123", to_string(123u));
  EXPECT_EQ("4294967295",
            to_string(std::numeric_limits<std::uint32_t>::max()));
#ifdef PRIMITIV_WORDSIZE_64
  EXPECT_EQ("0", to_string(0ll));
  EXPECT_EQ("5", to_string(5ll));
  EXPECT_EQ("123", to_string(123ll));
  EXPECT_EQ("9223372036854775807",
            to_string(std::numeric_limits<std::int64_t>::max()));
  EXPECT_EQ("-9223372036854775808",
            to_string(std::numeric_limits<std::int64_t>::min()));
  EXPECT_EQ("0", to_string(0ull));
  EXPECT_EQ("5", to_string(5ull));
  EXPECT_EQ("123", to_string(123ull));
  EXPECT_EQ("18446744073709551615",
            to_string(std::numeric_limits<std::uint64_t>::max()));
#endif  // PRIMITIV_WORDSIZE_64
  EXPECT_EQ("0.000000", to_string(0.0f));
  EXPECT_EQ("1.000000", to_string(1.0f));
  EXPECT_EQ("1.100000", to_string(1.1f));
  EXPECT_EQ("1.234568", to_string(1.23456789f));
  EXPECT_EQ("-1.000000", to_string(-1.0f));
  EXPECT_EQ("-1.100000", to_string(-1.1f));
  EXPECT_EQ("-12.345679", to_string(-12.3456789f));
  EXPECT_EQ("-340282346638528859811704183484516925440.000000",
            to_string(-std::numeric_limits<float>::max()));
  EXPECT_EQ("0.000000", to_string(0.0));
  EXPECT_EQ("1.000000", to_string(1.0));
  EXPECT_EQ("1.100000", to_string(1.1));
  EXPECT_EQ("1.234568", to_string(1.23456789));
  EXPECT_EQ("-1.000000", to_string(-1.0));
  EXPECT_EQ("-1.100000", to_string(-1.1));
  EXPECT_EQ("-12.345679", to_string(-12.3456789));
  EXPECT_EQ("-17976931348623157081452742373170435679807056752584499659891747"
             "68031572607800285387605895586327668781715404589535143824642343"
             "21326889464182768467546703537516986049910576551282076245490090"
             "38932894407586850845513394230458323690322294816580855933212334"
             "8274797826204144723168738177180919299881250404026184124858368."
             "000000",
            to_string(-std::numeric_limits<double>::max()));
#ifdef PRIMITIV_WORDSIZE_64
  EXPECT_EQ("0.000000", to_string(0.0L));
  EXPECT_EQ("1.000000", to_string(1.0L));
  EXPECT_EQ("1.100000", to_string(1.1L));
  EXPECT_EQ("1.234568", to_string(1.23456789L));
  EXPECT_EQ("-1.000000", to_string(-1.0L));
  EXPECT_EQ("-1.100000", to_string(-1.1L));
  EXPECT_EQ("-12.345679", to_string(-12.3456789L));
  EXPECT_EQ("-118973149535723176502126385303097020516906332229462420044032373"
            "389173700552297072261641029033652888285354569780749557731442744"
            "315367028843419812557385374367867359320070697326320191591828296"
            "152436552951064679108661431179063216977883889613478656060039914"
            "875343321145491116008867984515486651285234014977303760000912547"
            "939396622315138362241783854274391783813871780588948754057516822"
            "634765923557697480511372564902088485522249479139937758502601177"
            "354918009979622602685950855888360815984690023564513234659447638"
            "493985927645628457966177293040780660922910271504608538808795932"
            "778162298682754783076808004015069494230341172895777710033571401"
            "055977524212405734700738625166011082837911962300846927720096515"
            "350020847447079244384854591288672300061908512647211195136146752"
            "763351956292759795725027800298079590419313960302147099703527646"
            "744553092202267965628099149823208332964124103850923918473478612"
            "192169721054348428704835340811304257300221642134891734717423480"
            "071488075100206439051723424765600472176809648610799494341570347"
            "632064355862420744350442438056613601760883747816538902780957697"
            "597728686007148702828795556714140463261583262360276289631617397"
            "848425448686060994827086796804807870251185893083854658422304090"
            "880599629459458620190376604844679092600222541053077590106576067"
            "134720012584640695703025713896098375799892695455305236856075868"
            "317922311363951946885088077187210470520395758748001314313144425"
            "494391994017575316933939236688185618912993172910425292123683515"
            "992232205099800167710278403536014082929639811512287776813570604"
            "578934353545169653956125404884644716978689321167108722908808277"
            "835051822885764606221873970285165508372099234948333443522898475"
            "123275372663606621390228126470623407535207172405866507951821730"
            "346378263135339370677490195019784169044182473806316282858685774"
            "143258116536404021840272491339332094921949842244273042701987304"
            "453662035026238695780468200360144729199712309553005720614186697"
            "485284685618651483271597448120312194675168637934309618961510733"
            "006555242148519520176285859509105183947250286387163249416761380"
            "499631979144187025430270675849519200883791516940158174004671147"
            "787720145964446117520405945350476472180797576111172084627363927"
            "960033967047003761337450955318415007379641260504792325166135484"
            "129188421134082301547330475406707281876350361733290800595189632"
            "520707167390454777712968226520622565143991937680440029238090311"
            "243791261477625596469422198137514696707944687035800439250765945"
            "161837981185939204954403611491531078225107269148697980924094677"
            "214272701240437718740921675661363493890045123235166814608932240"
            "069799317601780533819184998193300841098599393876029260139091141"
            "452600372028487213241195542428210183120421610446740462163533690"
            "058366460659115629876474552506814500393294140413149540067760295"
            "100596225302282300363147382468105964844244132486457313743759509"
            "641616804802412935187620466813563687753281467553879887177183651"
            "289394719533506188500326760735438867336800207438784965701457609"
            "034985757124304510203873049485425670247933932280911052604153852"
            "899484920399109194612991249163328991799809438033787952209313146"
            "694614970593966415237594928589096048991612194498998638483702248"
            "667224914892467841020618336462741696957630763248023558797524525"
            "373703543388296086275342774001633343405508353704850737454481975"
            "472222897528108302089868263302028525992308416805453968791141829"
            "762998896457648276528750456285492426516521775079951625966922911"
            "497778896235667095662713848201819134832168799586365263762097828"
            "507009933729439678463987902491451422274252700636394232799848397"
            "673998715441855420156224415492665301451550468548925862027608576"
            "183712976335876121538256512963353814166394951655600026415918655"
            "485005705261143195291991880795452239464962763563017858089669222"
            "640623538289853586759599064700838568712381032959192649484625076"
            "899225841930548076362021508902214922052806984201835084058693849"
            "381549890944546197789302911357651677540623227829831403347327660"
            "395223160342282471752818181884430488092132193355086987339586127"
            "607367086665237555567580317149010847732009642431878007000879734"
            "603290627894355374356444885190719161645514115576193939969076741"
            "515640282654366402676009508752394550734155613586793306603174472"
            "092444651353236664764973540085196704077110364053815007348689179"
            "836404957060618953500508984091382686953509006678332447257871219"
            "660441528492484004185093281190896363417573989716659600075948780"
            "061916409485433875852065711654107226099628815012314437794400874"
            "930194474433078438899570184271000480830501217712356062289507626"
            "904285680004771889315808935851559386317665294808903126774702966"
            "254511086154895839508779675546413794489596052797520987481383976"
            "257859210575628440175934932416214833956535018919681138909184379"
            "573470326940634289008780584694035245347939808067427323629788710"
            "086717580253156130235606487870925986528841635097252953709111431"
            "720488774740553905400942537542411931794417513706468964386151771"
            "884986701034153254238591108962471088538580868883777725864856414"
            "593426212108664758848926003176234596076950884914966244415660441"
            "9552086811989770240.000000",
            to_string(-std::numeric_limits<long double>::max()));
#endif  // PRIMITIV_WORDSIZE_64
}

}  // namespace string_utils
}  // namespace primitiv
