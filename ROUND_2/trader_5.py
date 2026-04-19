"""Round 2 trader v5: trader_4 + dynamic MAKER_PENNY from a small OLS model.

WHY v5
──────
trader_4 used a constant `MAKER_PENNY = 1`. v5 keeps every other line
of trader_4 untouched and only changes the penny size based on a
spread forecast from a tiny OLS regression (replaces the CNN that the
original plan called for; same input/target structure, but ~1 KB of
weights instead of ~170 KB).

Model (offline-trained in train_osmium_ols.py, weights embedded below):
    input   x = [ mid_norm[t-W:t] ‖ spr_norm[t-W:t] ]    ∈ R^{2W}
    output  ŷ = x @ β + α                                ∈ R^2
            ŷ = (Δmid_norm_pred, spread_norm_pred) at t + HORIZON

    Per-window normalization (must match the trainer):
        mid_norm = (mid - mid[t-1]) / max(std(mid_window), 1e-3)
        spr_norm = (spread - mean(spread_window)) / max(std(spread_window), 1e-3)

    The spread head is de-normalized back to ticks before use.

Decisions
─────────
1. Penny size from predicted spread (widened from the plan after the
   first round of measurements: spreads on this book sit at ~16
   constantly, so the original ≤5/≤12 thresholds picked penny=3 every
   tick and gave up edge vs trader_4):

       pred_spread ≤ 15   → MAKER_PENNY = 1
       pred_spread ≤ 20   → MAKER_PENNY = 2
       pred_spread > 20   → MAKER_PENNY = 3

2. Directional quote shift from predicted Δmid (the OLS mid-delta head
   has real lift over naive on val, ~44%). When the model expects mid
   to rise we lift both quotes by 1 tick (and vice-versa). Capped at
   ±MID_SKEW_CAP ticks so it never overrides the inventory skew or the
   anti-cross guards.

Pepper logic, bid(), anti-cross guard, and inventory skew are kept
exactly as in trader_4.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ── Product names ────────────────────────────────────────────────────────
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

# ── PEPPER parameters (inherited from trader_4) ─────────────────────────
PEPPER_SLOPE = 0.001
PEPPER_TAKE_BUFFER = 8
PEPPER_OUTLIER_EDGE = 10

# ── OSMIUM parameters ──────────────────────────────────────────────────
FAIR_EMA_N = 30
FAIR_ALPHA = 2.0 / (FAIR_EMA_N + 1.0)

# trader_4 fallback for cold start / numpy unavailable / forward error.
MAKER_PENNY_FALLBACK = 1

SKEW_DIV = 12
MAX_MAKE_SIZE = 80
NARROW_SPREAD = 4

# Directional skew: tick adjustment derived from the predicted mid
# move. `pred_ticks * MID_SKEW_GAIN` is rounded then clipped to
# ±MID_SKEW_CAP. Conservative defaults; raise GAIN to lean harder on
# the prediction.
MID_SKEW_GAIN = 0.5
MID_SKEW_CAP = 1
# Only react to predictions of meaningful size (in ticks).
MID_SKEW_DEADBAND = 0.5

# ── BID for Round-2 side mechanic ──────────────────────────────────────
BID_VALUE = 30

# ── Embedded OLS model (produced by train_osmium_ols.py) ───────────────
_WEIGHTS_JSON = r'''{"arch":{"window":128,"horizon":8,"channels":2,"outputs":2},"weights":{"beta":[[-0.018885679609244914,0.017884874654872932],[-0.0014290039032691887,0.012334610960437508],[-0.009996734367953092,-0.0031819089334173896],[0.0014090461054068826,-0.008163991761006718],[0.00010069687596132249,-0.002440744135072128],[0.014931606484981623,-0.0020721493989370697],[-0.011059967915435166,0.015051782290953872],[0.013036290609457988,-0.0019643911818751263],[-0.012100005072449002,0.003874478799805597],[0.0033989248533074523,0.0036005997069889616],[0.011136962820673663,-0.008254818186433768],[0.014665912036768756,-0.00123566353318418],[-0.008373976267845773,-0.004610614159502173],[0.0013426945204943368,-0.004642714097015456],[-0.016137783236243954,0.00034827350508472867],[0.0031664284723493504,0.003215367743895947],[-0.010861535986715992,-0.016499914903605483],[-0.006205668764227199,-0.003939240446764738],[-0.0026885698864449444,0.008969275777001025],[0.0009639076855275368,0.0077911433189429315],[0.008484592770266352,0.0013564428699642751],[0.008574174829305892,0.007243178188456245],[0.007823428183932309,0.00480970878578167],[0.020063661429487633,0.0027402911789358225],[0.0009464852311255413,-0.01651016637554937],[-0.004645050663878963,-0.015498740880945246],[0.009773378165801951,-0.008658744457256999],[0.0003645184181519521,0.00824541086097606],[0.006788038973311568,0.005632273443834999],[-0.008907005128542539,-0.003957865821833822],[-0.002752457671818998,0.005684571555917225],[0.008248918033832306,0.005372512369156121],[0.0006822916180815826,-0.002507547251568642],[0.005901429346399512,0.010566505052757821],[-0.00784875271628612,0.009163225032884637],[0.013807604583986851,-0.011949617049628542],[-0.0003579016483814425,-0.009633841255952937],[0.011942212332726607,0.0021250121990044864],[0.009544631663839575,0.015165087746864996],[-0.012102144633484666,-0.02053360499062869],[-0.008108360589633962,0.020514409844795294],[0.0009517912329362814,-0.014209522267496165],[-0.006236106864961193,-0.005998038230247574],[0.00805452515038603,0.020196221561545744],[0.005722920312001302,-0.02280964334099116],[0.009550128707995514,0.00038474560037547896],[-0.0122271088466186,-0.016627669162286023],[-0.0066594793564443994,0.02036570890537214],[-0.012491678447380913,-0.0024194898502215545],[-0.010128050532356695,-0.010144279761622533],[0.012300551591898183,0.008814372804374462],[-0.009707214394787531,-0.0058017237490433715],[-0.005798075960715417,0.0015153854950484922],[-0.005383082388883084,-0.003967543581564086],[-0.00303419568783155,-0.0012931441503240024],[0.0029977176322087636,0.004007725568148847],[0.0004171269591091202,-0.0060410871716643595],[-0.004661899878117041,0.0006504867690885495],[-0.01039200823409518,0.0008684610313589128],[-0.005570119274300288,-0.002932087935835274],[-0.0036804356028450687,0.014556176663469896],[0.023575874588804124,0.0041294133298655986],[0.0075639797632528,0.019598788573308615],[-0.002562744182026749,-0.007254904940944099],[0.009793377671005648,-0.0003582628672852413],[0.00042017046876537025,0.007785004373523021],[-0.0002988480071851038,0.0031738406077732578],[0.011289118929442522,-0.008629545014728821],[-0.0021815534974098352,-0.013028269158394053],[-0.006696151451453435,-0.005502572539626667],[-0.0019910543599508095,0.006088718937753511],[0.0002969717633255445,-0.002744084672440216],[0.008211139522783482,-0.004028379558318857],[-0.003803829046657748,-0.008799217332262837],[-0.005166955680438578,0.009563717987521356],[0.009410852328457642,-0.0043441682497008735],[-0.0039017406855163346,-0.006254699867388472],[-0.0011059561051597353,-0.002670711889785324],[0.010744930435526075,0.027783517804883827],[-0.011783908244076977,-0.0085757423899361],[0.0038724531926677087,-0.006907992223328691],[-0.009271278752108899,-0.006828530176003071],[0.00028434168750005734,0.005875280290323639],[0.023702872484334468,0.014605936246196223],[0.0005202482971016378,-0.021408828147316445],[0.0006449848582110715,0.001223356883040682],[0.016959629081464467,-0.0063706753423708575],[-0.006322138069602401,-0.004550887139990842],[-0.0012476345798201683,0.025527664557517767],[-0.0033940649663245968,0.016179896172806126],[-0.0027853734496566538,-0.00345940592503338],[0.002986511371834275,-0.033581978083510086],[-0.004854782838022312,0.011511723626960482],[-0.016471574066086275,0.00016056193513897847],[0.017609636864369654,0.008230045756458592],[-0.00771260956051495,-0.012822756828209289],[-0.009953510748180584,-0.0058678735883373944],[0.008536069210768788,0.004150790368414036],[-0.006928266942591424,-0.017026993759641635],[-0.002608835375002743,-0.00013756480715565277],[-0.008654124433362656,0.007934452012392523],[0.002202026964314805,-0.005483564680832564],[0.008320783723867343,0.001960216196468179],[-0.009845425512894936,0.005014260944204658],[0.012717068498415935,-0.0001039626319408528],[0.0023308481984028887,0.0050948062907909195],[-0.0010553985505338491,-0.004527335652687315],[-0.013312149916915587,-0.010065729537731445],[0.008029017109454322,0.007270211567139547],[0.009600370587744067,-0.0014988916803837474],[0.003481310392755994,0.0036359645298180733],[-0.002829660295035255,-0.004502223245357728],[-0.009994026483361099,-0.0018158078261657513],[0.011845545960783008,0.007580519884280206],[0.011046828569030312,0.011793997772716166],[0.01058355756935106,0.0012656813603662586],[0.018907114593199395,-0.004996195404322706],[0.01697377486554425,0.007413672225482946],[0.030418623600347344,0.0024288654583396324],[0.02164631300840307,-0.008199808388536537],[0.03847328486522413,-0.004521430364090585],[0.05794345215771701,0.0003560074633312992],[0.0714041410162704,0.0005500257602148698],[0.07988336460992888,0.005484689476676702],[0.09242826774290161,0.0025385109463337344],[0.11185321992791322,0.012927678513179703],[0.17599509554362894,-0.0023876169419215117],[0.0,0.0],[-0.002168942286267743,0.0032752010691524643],[-0.001366742892618314,0.009093356698024814],[-0.004198337637401421,-0.0012900837559653897],[0.0026329713634927307,0.011531500296629986],[0.00622736666104825,-0.0036546201844419733],[-0.010033875266333264,0.0015656514478848116],[-0.005411376269514628,-0.005743409452743174],[-0.012517281545369149,0.007479597236807945],[0.006203316103681092,-0.008739520718154597],[0.0012321128840567426,-0.00695435656422431],[-0.001859764692075721,0.0009333180227386407],[-0.002683737046453807,-0.00836857400878232],[-0.007228790273303565,0.00994879304674209],[-0.008910262010721695,-0.0028901721762057114],[0.004329014914680933,-0.0033609486431492716],[0.0009090428327760694,-0.0033994406934968226],[0.022596525015528956,-0.00705396081195789],[-0.00833515923054472,-0.0049841931110405245],[-0.007879169174800364,-0.007599608512777975],[0.008033429881776898,-0.0011225085506933053],[-0.0020425286095090847,-0.011254139405881521],[-0.01589254464354176,-0.012075799909286545],[0.0019106970164407917,-0.007391243993944475],[0.0019788350659036952,0.0008179538635567454],[0.0034329091457535777,-0.0034628021277984013],[-0.005855742976283761,0.004639441270687287],[-0.0052554645375206465,-0.0021200218952649877],[0.007101903068069493,-0.0012792718091881375],[-0.006002604644698955,0.009056783047705219],[0.012811352387262567,0.017695794977218092],[0.005643100721180366,0.008578667347264637],[0.0006228003497077946,0.0008486165051895375],[0.002613836267889574,-0.0008363887557064736],[0.01206439919174534,-0.000461959416279277],[-0.002439190529920004,-0.01460331106594864],[0.0059231969589511055,0.011496056000158133],[0.00548613062729537,-0.0029884014969623705],[0.006447440242477926,-0.01316997875855107],[-0.008003834582417607,-0.011798377225564979],[0.0030660946864579214,-0.0036948079239492782],[-0.0048697360273902745,-0.010010660045331939],[0.001216123936877199,-0.004217409675336459],[-0.00807171594508166,0.003844110245134031],[-0.011232601372633625,0.0008785134605361476],[0.004788842600697311,0.01055135276283144],[0.0034605147753457867,-0.01507622061979077],[-0.005991149705150528,0.013306588839472987],[-0.006445194998293181,-0.00884466330957345],[0.002357579784097473,0.01832989909841883],[-0.01414030700710526,-0.009697829775597876],[-0.004032818268729458,-0.005835233379562926],[-0.01049920205375779,-0.0012167397597800719],[-0.00427637463594621,0.003947059360885299],[-0.001918355689111535,-0.0029472292416339797],[-0.009966362027181253,-0.0069629904390755785],[-0.003462453937934048,0.006849663775572561],[-0.007808500037571497,0.009812012411420218],[-0.010183324635637573,0.008994045244893353],[-0.0006872685567794926,0.005557887249547301],[-0.00915746723248497,0.008958589350201098],[-0.007449299683419591,0.0036503476399984587],[0.003118738654278444,-0.013124596010651083],[-0.00579023997256647,0.0002464800161944422],[0.005901690160370492,-0.00029409867672268095],[0.0021619789608618885,0.007520659990510196],[-0.009308927491719878,0.006594291643146027],[0.0024803203064320835,0.0105102408225677],[0.008788863987154623,-0.006353725509866111],[0.005325948196525002,0.009904303277935246],[0.0042168016502102165,-0.005980196153867265],[0.0027299317159136883,0.007861173185116439],[-0.0057025747827761444,-0.012486712151242637],[-0.007435990727576816,-0.0011208233625227887],[-0.0029083115783910423,0.012820011234146162],[-0.0021111256351122585,0.004993887224966466],[-0.004556050787170543,-0.014002108237054433],[-0.006831721081981754,-0.007342948007276642],[-0.006953410487011413,0.006172706501061652],[0.0030971768397541153,0.004169266554336867],[-0.0010317030756775687,-0.0072820071323709944],[0.009722652307527058,-0.004420642421566894],[0.004031465055978253,0.0028101071649383833],[-0.0035786365380810355,-0.0038916416745820555],[0.004598476998875946,-0.0007727233684916218],[0.013693028588212868,0.002823268740233576],[0.009928745475343921,0.0078749291399671],[0.011506096909956905,0.014609511597374466],[0.024166282430403738,0.007841550793550759],[0.001750721532542926,-0.0022875992701503656],[0.0063679813940955765,0.0005584011640017674],[0.004639494467920758,-0.0008493998570921875],[0.00596026754253579,-0.011930185698622312],[0.002731752005183394,0.006866619759343642],[0.009735327072723917,0.01263876218215965],[-0.007247448255343139,-0.004199643947530137],[0.007369569603039471,-0.011411012984287332],[0.005260555744528174,0.005227501820341374],[0.0044950782596439075,-0.002121454482762894],[0.004255311303750612,-0.004049989939491374],[0.0034517081222654547,-0.0005367670072423671],[0.003310588243318484,-0.0015084194541056738],[0.005335579023225042,0.0008280581487991923],[-0.009463366767383515,-0.01685903823387028],[0.012766287185386882,0.0040535976494756685],[0.007758539319711596,-0.010547840317719746],[-0.006644278131746113,-0.00459993591107317],[-0.002888800794640345,0.013605948000489088],[0.0005722409219346907,0.01159826624160884],[-0.002522102080059484,0.006386880295433734],[0.002359272971893833,-0.000728412711332469],[0.0003447955881628686,-0.002388979128678848],[0.002880300278289069,0.00042992489082113527],[0.006045715625114413,0.001015117480194414],[-0.003294696663343542,-0.008551583666208338],[0.0025384110560517084,-0.012474738991146709],[0.005100828212585615,0.0005393803901460789],[0.010462873509049026,-0.00569266411714497],[-0.0008265796103889912,-0.010030054558643559],[0.0037937183391899567,-0.0017065522856301227],[-0.003905831824503431,-0.007865578614361416],[-0.004764171434862314,0.01596699102019125],[-0.004111951366330632,0.007466422216261866],[-0.00867188621765013,0.0009598759470018658],[0.006996451555011964,0.007020335711995948],[0.0029210387982679304,0.012900307395670552],[-0.0009443038055032891,-0.0013006650759935401],[-0.012010412447853494,0.007007803452246399],[-0.015969017179055457,0.0003791749406704899]],"alpha":[0.0007472135333295052,-0.007586707814353938]},"train":{"ridge":1e-06,"days_train":[-1,0],"days_val":[1],"val_mid_mse":0.9268005061122869,"val_spr_mse":1.097933973112567,"naive_val_mid_mse":1.3064426800869982,"naive_val_spr_mse":1.086444710538845,"lift_spr_pct":-1.0575101026561782}}'''

_MODEL = json.loads(_WEIGHTS_JSON)
_ARCH = _MODEL["arch"]
_W = _MODEL["weights"]

_WINDOW = _ARCH["window"]
_HORIZON = _ARCH["horizon"]

if _HAS_NUMPY:
    _BETA = np.asarray(_W["beta"], dtype=np.float64)    # (2W, 2)
    _ALPHA = np.asarray(_W["alpha"], dtype=np.float64)  # (2,)


# ── numpy-only forward pass (matmul + bias) ────────────────────────────
def _forward(mid_hist, spr_hist):
    """Run the OLS model on one window.

    Returns (pred_mid_delta_ticks, pred_spread_ticks) — both in *tick*
    units (already de-normalized), ready for the decision logic.
    """
    mid = np.asarray(mid_hist, dtype=np.float64)
    spr = np.asarray(spr_hist, dtype=np.float64)

    ref_mid = float(mid[-1])
    mid_std = max(float(mid.std()), 1e-3)
    mid_norm = (mid - ref_mid) / mid_std

    spr_mean = float(spr.mean())
    spr_std = max(float(spr.std()), 1e-3)
    spr_norm = (spr - spr_mean) / spr_std

    x = np.concatenate([mid_norm, spr_norm])    # (2W,)
    y = x @ _BETA + _ALPHA                      # (2,)

    pred_mid_delta = float(y[0]) * mid_std
    pred_spread = float(y[1]) * spr_std + spr_mean
    return pred_mid_delta, pred_spread


def _pick_penny(pred_spread: float) -> int:
    """Dynamic maker-penny from predicted spread.

    Thresholds widened after live measurement: this book's spread sits
    near 16 nearly always, so under the original ≤5/≤12 cuts we'd
    pay penny=3 forever and bleed edge vs trader_4.
    """
    if pred_spread <= 15.0:
        return 1
    if pred_spread <= 20.0:
        return 2
    return 3


def _dir_shift(pred_mid_delta: float) -> int:
    """Tick adjustment from the predicted mid move.

    Positive = mid expected to rise → lift both quotes; negative =
    drop both quotes. Deadbanded so we don't react to tiny noise, then
    clipped to ±MID_SKEW_CAP.
    """
    if abs(pred_mid_delta) < MID_SKEW_DEADBAND:
        return 0
    raw = round(pred_mid_delta * MID_SKEW_GAIN)
    if raw > MID_SKEW_CAP:
        return MID_SKEW_CAP
    if raw < -MID_SKEW_CAP:
        return -MID_SKEW_CAP
    return int(raw)


class Trader:
    # ── Round-2 sealed bid ────────────────────────────────────────────
    def bid(self) -> int:
        return BID_VALUE

    # ── main entry ────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        data = self._deserialize(state.traderData)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state, data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, data)

        return result, 0, self._serialize(data)

    # ── state helpers ─────────────────────────────────────────────────
    def _deserialize(self, raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _serialize(self, data: dict) -> str:
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            return ""

    # ── INTARIAN_PEPPER_ROOT (inherited from trader_4) ────────────────
    def _trade_pepper(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[PEPPER]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        if "pepper_init_mid" not in data:
            data["pepper_init_mid"] = (best_bid + best_ask) / 2.0

        fair = data["pepper_init_mid"] + PEPPER_SLOPE * state.timestamp
        position = state.position.get(PEPPER, 0)

        bought = 0
        for ask in sorted(depth.sell_orders.keys()):
            if ask > fair + PEPPER_TAKE_BUFFER:
                break
            room = POSITION_LIMIT - position - bought
            if room <= 0:
                break
            ask_vol = -depth.sell_orders[ask]
            qty = min(ask_vol, room)
            if qty > 0:
                orders.append(Order(PEPPER, int(ask), qty))
                bought += qty

        room = POSITION_LIMIT - position - bought
        if room > 0:
            orders.append(Order(PEPPER, int(math.floor(fair)), room))

        net_long = position + bought
        if net_long > 0:
            orders.append(
                Order(PEPPER, int(math.ceil(fair)) + PEPPER_OUTLIER_EDGE, -net_long)
            )

        return orders

    # ── ASH_COATED_OSMIUM: penny-jump maker w/ OLS-driven penny size ──
    def _trade_osmium(self, state: TradingState, data: dict) -> list[Order]:
        orders: list[Order] = []
        depth: OrderDepth = state.order_depths[OSMIUM]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        prev = data.get("osmium_fair")
        fair = mid if prev is None else FAIR_ALPHA * mid + (1.0 - FAIR_ALPHA) * prev
        data["osmium_fair"] = fair
        fair_int = int(round(fair))

        # Rolling buffers for the OLS forecast.
        mid_hist = data.get("osmium_mid_hist") or []
        spr_hist = data.get("osmium_spr_hist") or []
        mid_hist.append(mid)
        spr_hist.append(spread)
        if len(mid_hist) > _WINDOW:
            mid_hist = mid_hist[-_WINDOW:]
            spr_hist = spr_hist[-_WINDOW:]
        data["osmium_mid_hist"] = mid_hist
        data["osmium_spr_hist"] = spr_hist

        # OLS forecast → dynamic penny size + directional tick shift.
        # Cold start (or numpy missing / forward error) falls back to
        # trader_4's static behavior.
        maker_penny = MAKER_PENNY_FALLBACK
        dir_shift = 0
        if _HAS_NUMPY and len(mid_hist) >= _WINDOW:
            try:
                pred_mid_delta, pred_spread = _forward(mid_hist, spr_hist)
                maker_penny = _pick_penny(pred_spread)
                dir_shift = _dir_shift(pred_mid_delta)
                data["osmium_last_pred_spread"] = pred_spread
                data["osmium_last_pred_mid_delta"] = pred_mid_delta
                data["osmium_last_penny"] = maker_penny
                data["osmium_last_dir_shift"] = dir_shift
            except Exception:
                maker_penny = MAKER_PENNY_FALLBACK
                dir_shift = 0

        position = state.position.get(OSMIUM, 0)
        skew = position // SKEW_DIV

        if spread <= NARROW_SPREAD:
            bid_px = fair_int - 1 - skew + dir_shift
            ask_px = fair_int + 1 - skew + dir_shift
        else:
            bid_px = best_bid + maker_penny - skew + dir_shift
            ask_px = best_ask - maker_penny - skew + dir_shift

        # Anti-cross guard: never quote across the fair value.
        bid_px = min(bid_px, fair_int - 1)
        ask_px = max(ask_px, fair_int + 1)

        # Anti-self-cross.
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        buy_size = min(buy_room, MAX_MAKE_SIZE)
        sell_size = min(sell_room, MAX_MAKE_SIZE)

        if buy_size > 0:
            orders.append(Order(OSMIUM, int(bid_px), buy_size))
        if sell_size > 0:
            orders.append(Order(OSMIUM, int(ask_px), -sell_size))

        return orders
