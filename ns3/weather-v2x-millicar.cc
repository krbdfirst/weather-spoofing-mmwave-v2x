/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// weather-v2x-millicar.cc: weather-spoofing band-selection attack on an NR V2X
// sidelink platoon, modelled at mmWave with MilliCar (ns-3.36.1 + ns3-mmwave).
// The FIS-selected band is the carrier frequency of the simulated link, so 3GPP
// V2V path loss, beamforming and blockage come from the native channel model.
// Only rain attenuation is added (ITU-R P.838 / Dataset-1), because the 3GPP
// channel models have no rain term.
//
// Scenarios:  baseline (FIS picks band from rain), attackA (force up 73 GHz),
//             attackB (force down 5 GHz).
//
// Two bandwidth regimes (--regime):
//   fixwidth   100 MHz for every band (MilliCar default). The carrier frequency
//              is the only difference across arms, so capacity is held constant.
//   realwidth  each band uses a representative deployable width (5.9 GHz ITS
//              20 MHz, mmWave up to the NR maximum of 400 MHz), so a band switch
//              changes both reach and capacity. Bandwidth does not change
//              propagation; it sets the thermal-noise floor (N=kTB) and the
//              transport-block budget.
//
// Platoon of <= 8 vehicles (MilliCar group cap = SlotsPerSubframe = 2^numerology),
// full-mesh CAM exchange. Output logs:
//   pkts-<tag>.csv  txrx,time,src,dst,seq,bytes,latency_ms,dist_m,x,y   (per packet)
//   phy-<tag>.csv   time,rxnode,txnode,dist_m,sinr_db,numsym,tbsize,mcs (per rx TB)
//   meta-<tag>.csv  regime,scenario,rain,band_ghz,bw_mhz,numveh,simtime (one row)

#include "ns3/mmwave-sidelink-spectrum-phy.h"
#include "ns3/mmwave-vehicular-net-device.h"
#include "ns3/mmwave-vehicular-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/ns2-mobility-helper.h"
#include "ns3/config.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/buildings-module.h"
#include "ns3/command-line.h"
#include "ns3/node-list.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/propagation-module.h"
#include "ns3/seq-ts-header.h"
#include "ns3/seq-ts-size-header.h"
#include "ns3/spectrum-value.h"

#include <fstream>
#include <map>
#include <vector>
#include <cmath>

using namespace ns3;
using namespace millicar;

NS_LOG_COMPONENT_DEFINE ("WeatherV2xMilliCar");

// Weather / FIS model (Dataset-1, ITU-R P.838)
// Per-band specific attenuation γ (dB/km), applied per link as γ·d (rain only).
static double
LookupExtraAttenuationDb (double rainMmHr, double freqGHz)
{
  static const double kRain[20] = {0.0000,  2.6316,  5.2632,  7.8947,  10.5263,
                                   13.1579, 15.7895, 18.4211, 21.0526, 23.6842,
                                   26.3158, 28.9474, 31.5789, 34.2105, 36.8421,
                                   39.4737, 42.1053, 44.7368, 47.3684, 50.0000};
  static const double kExtra_5GHz[20]  = {0.0000, 0.0011, 0.0036, 0.0072, 0.0118,
                                          0.0172, 0.0234, 0.0304, 0.0381, 0.0465,
                                          0.0556, 0.0653, 0.0757, 0.0867, 0.0984,
                                          0.1106, 0.1234, 0.1367, 0.1507, 0.1651};
  static const double kExtra_28GHz[20] = {0.0000, 0.5232, 1.0234, 1.5152, 2.0016,
                                          2.4842, 2.9636, 3.4405, 3.9151, 4.3879,
                                          4.8589, 5.3285, 5.7967, 6.2636, 6.7294,
                                          7.1941, 7.6578, 8.1206, 8.5825, 9.0436};
  static const double kExtra_39GHz[20] = {0.0000, 0.9822,  1.8005,  2.5666,  3.3006,
                                          4.0116, 4.7049,  5.3837,  6.0504,  6.7067,
                                          7.3539, 7.9929,  8.6247,  9.2499,  9.8691,
                                          10.4827, 11.0913, 11.6950, 12.2943, 12.8894};
  static const double kExtra_60GHz[20] = {0.0000,  1.8053,  3.0691,  4.1864,  5.2179,
                                          6.1900,  7.1173,  8.0089,  8.8710,  9.7082,
                                          10.5238, 11.3204, 12.1003, 12.8650, 13.6161,
                                          14.3546, 15.0818, 15.7983, 16.5050, 17.2026};
  static const double kExtra_73GHz[20] = {0.0000,  2.1748,  3.5993,  4.8329,  5.9568,
                                          7.0057,  7.9984,  8.9467,  9.8586,  10.7397,
                                          11.5945, 12.4262, 13.2374, 14.0304, 14.8069,
                                          15.5683, 16.3160, 17.0510, 17.7743, 18.4867};
  const double* col = kExtra_5GHz;
  if (freqGHz < 7.0)        col = kExtra_5GHz;
  else if (freqGHz < 33.5)  col = kExtra_28GHz;
  else if (freqGHz < 49.5)  col = kExtra_39GHz;
  else if (freqGHz < 66.5)  col = kExtra_60GHz;
  else                      col = kExtra_73GHz;

  const double R = std::max (0.0, rainMmHr);
  if (R <= 0.0) return 0.0;

  // Above the table's 50 mm/h top row, extrapolate with the ITU-R P.838 power law
  // gamma = k*R^alpha. A power-law regression of the table over R >= 10 mm/h fits
  // within 0.16% (0.00% for every mmWave band). The coefficients below are that
  // regression and are continuous with the table at R = 50. Extrapolation is
  // needed because an EXCELL cell calibrated to a 50 mm/h platoon mean can have a
  // peak well above 50 mm/h at its centre.
  if (R > 50.0)
    {
      double k, alpha;
      if      (freqGHz < 7.0)   { k = 0.00022; alpha = 1.6940; }
      else if (freqGHz < 33.5)  { k = 0.20509; alpha = 0.9679; }
      else if (freqGHz < 49.5)  { k = 0.42150; alpha = 0.8743; }
      else if (freqGHz < 66.5)  { k = 0.86061; alpha = 0.7656; }
      else                      { k = 1.07643; alpha = 0.7268; }
      return k * std::pow (R, alpha);
    }
  // Exactly 50 mm/h returns the top row directly, so uniform-mode runs are
  // bit-identical to Phase 1.
  if (R >= 50.0) return col[19];
  int lo = 0, hi = 18;
  while (lo < hi - 1)
    {
      const int mid = (lo + hi) / 2;
      if (kRain[mid] <= R) lo = mid; else hi = mid;
    }
  const double t = (R - kRain[lo]) / (kRain[lo + 1] - kRain[lo]);
  return col[lo] + t * (col[lo + 1] - col[lo]);
}

/// FIS band selection (Veins cascade): highest band whose base+rain attenuation
/// stays under the 74 dB budget; steps 73 -> 60 -> 39 -> 28 GHz.
static double
FisSelectBandGHz (double rainMmHr)
{
  constexpr double BUDGET = 74.0;
  if (69.93 + LookupExtraAttenuationDb (rainMmHr, 73.0) <= BUDGET) return 73.0;
  if (68.49 + LookupExtraAttenuationDb (rainMmHr, 60.0) <= BUDGET) return 60.0;
  if (65.35 + LookupExtraAttenuationDb (rainMmHr, 39.0) <= BUDGET) return 39.0;
  return 28.0;
}

// Rain field (Capsoni EXCELL + advection)
// Phase 1 used a single scalar rain rate: uniform over every link, constant for
// the whole run. Phase 2 adds a spatially and temporally varying field built from
// two established pieces:
//
//   EXCELL (Capsoni et al., Radio Science 22(3):395-404, 1987), eq (13):
//       R(x,y) = R_M * exp( -sqrt(x^2+y^2) / rho_0 )
//   with the cell defined down to its 5 mm/h contour, and the mean cell-size
//   parameter given by the double power law of eq (19), for R_M > 5:
//       rho_0_bar(R_M) = 1.7 * [ (R_M/6)^-1.0 + (R_M/6)^-0.26 ]   [km]
//
//   Frozen-field advection (the assumption underlying the synthetic storm
//   technique, Matricciani, Radio Science 31(2):281-295, 1996): the cell is a
//   rigid pattern translating at a constant storm speed, so
//       c(t) = c_0 + v_cell * t
//
// Attenuation is then the path integral of gamma(f, R(s,t)) along the link, so it
// varies per link and over time.

/// Capsoni eq (19): mean cell-size parameter (km) for a given peak rain rate.
static double
MeanCellRadiusKm (double peakMmHr)
{
  const double x = std::max (peakMmHr, 5.0) / 6.0;
  return 1.7 * (std::pow (x, -1.0) + std::pow (x, -0.26));
}

struct RainField
{
  enum Mode { UNIFORM, CELL };
  Mode mode {UNIFORM};
  double uniformRate {0.0};   // mm/h, UNIFORM mode
  double peakRate {0.0};      // R_M at the cell centre, mm/h
  double rho0Km {1.0};        // e-folding radius, km
  double cx0 {0.0}, cy0 {0.0};   // cell centre at t=0, metres
  double vx {0.0}, vy {0.0};     // advection velocity, m/s

  /// Rain rate (mm/h) at a point at time t.
  double RateAt (double px, double py, double t) const
  {
    if (mode == UNIFORM) return uniformRate;
    const double dx = px - (cx0 + vx * t);
    const double dy = py - (cy0 + vy * t);
    return peakRate * std::exp (-std::hypot (dx, dy) / (rho0Km * 1000.0));
  }
};

static RainField g_rain;              // the one field the whole run shares
static uint32_t g_pathSamples = 16;   // quadrature points along a link

/// Path-averaged specific attenuation (dB/km) over the segment a->b at time t.
/// Midpoint rule; for UNIFORM this collapses to the Phase 1 constant exactly.
static double
PathMeanGammaDbPerKm (double ax, double ay, double bx, double by,
                      double t, double bandGHz)
{
  if (g_rain.mode == RainField::UNIFORM)
    return LookupExtraAttenuationDb (g_rain.uniformRate, bandGHz);
  double acc = 0.0;
  for (uint32_t i = 0; i < g_pathSamples; ++i)
    {
      const double u = (i + 0.5) / g_pathSamples;
      acc += LookupExtraAttenuationDb (
               g_rain.RateAt (ax + u * (bx - ax), ay + u * (by - ay), t), bandGHz);
    }
  return acc / g_pathSamples;
}

/// Synthetic convoy: cumulative along-road offsets giving a 15..290 m spread.
static double
ConvoyY (uint32_t i)
{
  static const double cum[8] = {0, 15, 35, 60, 95, 140, 200, 290};
  return (i < 8) ? cum[i] : 290.0 + 60.0 * (i - 7);
}

/// Mean of exp(-r/rho_0) over every directed link, every path sample and the whole
/// run: the factor by which the platoon-experienced mean rain sits below the cell
/// peak. The EXCELL profile is linear in R_M, so calibrating the mean to a target
/// is one division rather than a search (though rho_0 itself depends on R_M via
/// eq (19), which is why the caller iterates).
static double
PlatoonShapeFactor (double rho0Km, double cx0, double cy0, double vx, double vy,
                    uint32_t numVeh, double speed, double simTime,
                    uint32_t timeSteps = 60)
{
  double acc = 0.0;
  uint32_t cnt = 0;
  for (uint32_t k = 0; k < timeSteps; ++k)
    {
      const double t = simTime * (k + 0.5) / timeSteps;
      const double ccx = cx0 + vx * t, ccy = cy0 + vy * t;
      for (uint32_t i = 0; i < numVeh; ++i)
        for (uint32_t j = 0; j < numVeh; ++j)
          {
            if (i == j) continue;
            const double yi = ConvoyY (i) + speed * t;
            const double yj = ConvoyY (j) + speed * t;
            for (uint32_t s = 0; s < g_pathSamples; ++s)
              {
                const double u = (s + 0.5) / g_pathSamples;
                const double py = yi + u * (yj - yi);
                acc += std::exp (-std::hypot (0.0 - ccx, py - ccy) / (rho0Km * 1000.0));
                ++cnt;
              }
          }
    }
  return cnt ? acc / cnt : 1.0;
}

/// Per-band representative channel width for the "realwidth" regime (Hz).
/// More spectrum is available higher up; capped at NR's 400 MHz max carrier.
///   5.9 GHz -> 20 MHz (C-V2X/802.11bd ITS band),  28 -> 100,  39 -> 200,
///   60/73   -> 400 (FR2 max).  Edit here to retune the regime.
static double
RealWidthHz (double bandGHz)
{
  if (bandGHz < 7.0)        return 20e6;
  else if (bandGHz < 33.5)  return 100e6;
  else if (bandGHz < 49.5)  return 200e6;
  else                      return 400e6;
}

// Rain-only per-link propagation loss
// Chained onto MilliCar's 3GPP V2V channel. Subtracts only the rain attenuation
// γ(band,rain)·d_km; the channel model provides the frequency-dependent path loss,
// beamforming and blockage.
class WeatherRainLossModel : public PropagationLossModel
{
public:
  static TypeId GetTypeId ();
  void SetBandGHz (double b) { m_bandGHz = b; }
private:
  // Excess loss is the line integral of the specific attenuation along the link,
  //   A_rain = integral of gamma(f, R(s,t)) ds,
  // evaluated against the shared rain field at the current simulation time. In
  // UNIFORM mode this reduces exactly to the Phase 1 expression gamma * d/1000.
  double DoCalcRxPower (double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
  {
    const Vector pa = a->GetPosition ();
    const Vector pb = b->GetPosition ();
    const double d = a->GetDistanceFrom (b);                       // metres
    const double gammaPerKm = PathMeanGammaDbPerKm (
      pa.x, pa.y, pb.x, pb.y, Simulator::Now ().GetSeconds (), m_bandGHz);
    return txPowerDbm - gammaPerKm * (d / 1000.0);
  }
  int64_t DoAssignStreams (int64_t /*stream*/) override { return 0; }
  double m_bandGHz {28.0};
};
NS_OBJECT_ENSURE_REGISTERED (WeatherRainLossModel);
TypeId
WeatherRainLossModel::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::WeatherRainLossModel")
                        .SetParent<PropagationLossModel> ()
                        .SetGroupName ("Propagation")
                        .AddConstructor<WeatherRainLossModel> ();
  return tid;
}

// Logging
static std::ofstream g_pkt;                          // per-packet tx/rx
static std::ofstream g_phy;                          // per-rx-TB SINR/MCS
static std::map<uint32_t, uint32_t> g_octetToNode;   // IP last octet -> node id
static NodeContainer g_nodes;                        // for position/distance lookups

static double
LinkDist (uint32_t a, uint32_t b)
{
  Vector pa = g_nodes.Get (a)->GetObject<MobilityModel> ()->GetPosition ();
  Vector pb = g_nodes.Get (b)->GetObject<MobilityModel> ()->GetPosition ();
  return std::hypot (pa.x - pb.x, pa.y - pb.y);
}

// tx/rx rows carry the directed flow (src,dst), sequence, payload bytes, the
// instantaneous link distance and the transmitter/receiver position. Latency is
// recorded on rx rows (now - send-timestamp from the SeqTsSize header); -1 on tx.
static void
LogTx (uint32_t src, uint32_t dst, Ptr<const Packet> /*p*/, const Address& /*from*/,
       const Address& /*to*/, const SeqTsSizeHeader& h)
{
  Vector pos = g_nodes.Get (src)->GetObject<MobilityModel> ()->GetPosition ();
  g_pkt << "tx," << Simulator::Now ().GetSeconds () << ',' << src << ',' << dst << ','
        << h.GetSeq () << ',' << h.GetSize () << ",-1," << LinkDist (src, dst) << ','
        << pos.x << ',' << pos.y << '\n';
}

static void
LogRx (uint32_t dst, Ptr<const Packet> /*p*/, const Address& from,
       const Address& /*to*/, const SeqTsSizeHeader& h)
{
  uint32_t octet = Ipv4Address::ConvertFrom (InetSocketAddress::ConvertFrom (from).GetIpv4 ()).Get () & 0xFF;
  uint32_t src = g_octetToNode.count (octet) ? g_octetToNode[octet] : 0;
  Vector pos = g_nodes.Get (dst)->GetObject<MobilityModel> ()->GetPosition ();
  double latMs = (Simulator::Now () - h.GetTs ()).GetSeconds () * 1000.0;
  g_pkt << "rx," << Simulator::Now ().GetSeconds () << ',' << src << ',' << dst << ','
        << h.GetSeq () << ',' << h.GetSize () << ',' << latMs << ',' << LinkDist (src, dst) << ','
        << pos.x << ',' << pos.y << '\n';
}

// Native MilliCar SINR/MCS report (fired per received transport block). rnti is
// the transmitter's RNTI; MilliCar assigns RNTI = nodeId + 1, so txnode = rnti-1.
static void
LogSinr (uint32_t rxNode, const SpectrumValue& sinr, uint16_t rnti, uint8_t numSym,
         uint32_t tbSize, uint8_t mcs)
{
  double sinrAvg = Sum (sinr) / sinr.GetSpectrumModel ()->GetNumBands ();
  uint32_t txNode = (rnti >= 1) ? (rnti - 1u) : 0u;
  g_phy << Simulator::Now ().GetSeconds () << ',' << rxNode << ',' << txNode << ','
        << LinkDist (rxNode, txNode) << ',' << 10.0 * std::log10 (sinrAvg) << ','
        << uint32_t (numSym) << ',' << tbSize << ',' << uint32_t (mcs) << '\n';
}

int
main (int argc, char* argv[])
{
  // parameters
  std::string scenario   = "baseline";       // baseline | attackA | attackB
  double trueRainRate    = 0.0;              // mm/hr
  uint32_t numVehicles   = 8;               // <= SlotsPerSubframe (2^numerology)
  double simTime         = 10.0;            // seconds
  std::string regime     = "fixwidth";      // fixwidth (100 MHz) | realwidth (per-band)
  double bandwidthHz     = -1.0;            // <=0 -> derive from regime; >0 -> override
  uint32_t numerology    = 3;
  double speed           = 20.0;            // m/s convoy speed
  double camIntervalMs   = 100.0;           // 10 Hz CAM
  uint32_t camBytes      = 200;
  std::string mobilityFile = "";            // optional SUMO ns-2 trace; empty -> convoy
  std::string simTag     = "default";
  std::string outputDir  = "./";
  bool logPkts           = false;
  uint32_t rngRun        = 1;               // channel RNG (acts as the seed)
  double forceUpGHz      = 73.0;
  double forceDownGHz    = 5.0;
  std::string rainField  = "uniform";       // uniform (Phase 1) | cell
  double cellSizeFactor  = 1.0;             // rho_0 / rho_0_bar(R_M), Capsoni eq (19)
  double cellSpeed       = 10.0;            // storm translation speed, m/s
  std::string cellDir    = "cross";         // along | against | cross
  uint32_t pathSamples   = 16;              // quadrature points per link
  bool calibrateOnly     = false;           // print field calibration and exit

  CommandLine cmd;
  cmd.AddValue ("scenario", "baseline | attackA (force up) | attackB (force down)", scenario);
  cmd.AddValue ("trueRainRate", "true ambient rain (mm/hr)", trueRainRate);
  cmd.AddValue ("numVehicles", "platoon size (<= 2^numerology)", numVehicles);
  cmd.AddValue ("simTime", "simulation time (s)", simTime);
  cmd.AddValue ("regime", "bandwidth regime: fixwidth (100 MHz) | realwidth (per-band)", regime);
  cmd.AddValue ("bandwidthHz", "channel bandwidth (Hz); <=0 derives from regime", bandwidthHz);
  cmd.AddValue ("speed", "convoy speed (m/s)", speed);
  cmd.AddValue ("mobilityFile", "optional SUMO ns-2 trace (empty = synthetic convoy)", mobilityFile);
  cmd.AddValue ("simTag", "tag for output files", simTag);
  cmd.AddValue ("outputDir", "output directory", outputDir);
  cmd.AddValue ("logPkts", "write per-packet CSV", logPkts);
  cmd.AddValue ("rngRun", "RNG run number (seed for channel realisation)", rngRun);
  cmd.AddValue ("rainField", "uniform (Phase 1) | cell (advecting EXCELL cell)", rainField);
  cmd.AddValue ("cellSizeFactor", "rho_0 as a multiple of Capsoni eq.(19) mean", cellSizeFactor);
  cmd.AddValue ("cellSpeed", "storm translation speed (m/s)", cellSpeed);
  cmd.AddValue ("cellDir", "cell drift: along | against | cross (rel. to road)", cellDir);
  cmd.AddValue ("pathSamples", "quadrature points along a link for the rain integral", pathSamples);
  cmd.AddValue ("calibrateOnly", "print the rain-field calibration and exit (no simulation)", calibrateOnly);
  cmd.Parse (argc, argv);

  RngSeedManager::SetRun (rngRun);
  g_pathSamples = std::max (1u, pathSamples);

  // resolve band -> carrier frequency
  double bandGHz = 28.0;
  if (scenario == "baseline")      bandGHz = FisSelectBandGHz (trueRainRate);
  else if (scenario == "attackA")  bandGHz = forceUpGHz;
  else if (scenario == "attackB")  bandGHz = forceDownGHz;
  else NS_FATAL_ERROR ("unknown scenario " << scenario);
  double frequency = bandGHz * 1e9;

  // resolve bandwidth from regime (unless explicitly overridden)
  if (regime != "fixwidth" && regime != "realwidth")
    NS_FATAL_ERROR ("unknown regime " << regime << " (fixwidth | realwidth)");
  double bwHz = bandwidthHz;
  if (bwHz <= 0.0)
    bwHz = (regime == "realwidth") ? RealWidthHz (bandGHz) : 100e6;

  // configure the rain field
  // Band selection uses the nominal rain rate in both modes. The weather input is a
  // broadcast regional report rather than an onboard point sensor, and the
  // selected band is therefore the same as in Phase 1; only the true field differs.
  if (rainField != "uniform" && rainField != "cell")
    NS_FATAL_ERROR ("unknown rainField " << rainField << " (uniform | cell)");
  g_rain.mode = (rainField == "cell") ? RainField::CELL : RainField::UNIFORM;
  g_rain.uniformRate = trueRainRate;

  if (g_rain.mode == RainField::CELL && trueRainRate > 0.0)
    {
      // Drift direction relative to the road (the convoy runs along +y).
      if (cellDir == "along")        { g_rain.vx = 0.0;       g_rain.vy =  cellSpeed; }
      else if (cellDir == "against") { g_rain.vx = 0.0;       g_rain.vy = -cellSpeed; }
      else if (cellDir == "cross")   { g_rain.vx = cellSpeed; g_rain.vy =  0.0; }
      else NS_FATAL_ERROR ("unknown cellDir " << cellDir << " (along|against|cross)");

      // The cell is centred on the platoon centroid at the midpoint of the run, so
      // it approaches, passes over and departs, and the rain rate rises and falls.
      double yc = 0.0;
      for (uint32_t i = 0; i < numVehicles; ++i) yc += ConvoyY (i);
      yc /= numVehicles;
      const double tMid = 0.5 * simTime;
      g_rain.cx0 = 0.0 - g_rain.vx * tMid;
      g_rain.cy0 = (yc + speed * tMid) - g_rain.vy * tMid;

      // Calibrate the peak so the platoon-experienced mean rain equals the nominal
      // sweep value, so uniform and cell runs at the same nominal rate carry the
      // same amount of rain. rho_0 depends on R_M through eq (19), so iterate to a
      // fixed point.
      double RM = trueRainRate, shape = 1.0;
      for (int it = 0; it < 12; ++it)
        {
          g_rain.rho0Km = cellSizeFactor * MeanCellRadiusKm (RM);
          shape = PlatoonShapeFactor (g_rain.rho0Km, g_rain.cx0, g_rain.cy0,
                                      g_rain.vx, g_rain.vy, numVehicles, speed, simTime);
          RM = trueRainRate / std::max (shape, 1e-6);
        }
      g_rain.peakRate = RM;
      std::cout << "rainField=cell rho0=" << g_rain.rho0Km << " km (factor "
                << cellSizeFactor << " x " << MeanCellRadiusKm (RM)
                << "), R_M=" << RM << " mm/hr, shape=" << shape
                << " -> platoon-mean rain=" << RM * shape << " mm/hr (target "
                << trueRainRate << "), drift=" << cellDir << " @ " << cellSpeed
                << " m/s" << std::endl;
    }

  if (calibrateOnly)
    {
      std::cout << "calibrateOnly: exiting before simulation" << std::endl;
      return 0;
    }

  std::cout << "regime=" << regime << " scenario=" << scenario << " rain=" << trueRainRate
            << " mm/hr -> band=" << bandGHz << " GHz (native carrier), bw="
            << bwHz / 1e6 << " MHz, rainGamma="
            << LookupExtraAttenuationDb (trueRainRate, bandGHz) << " dB/km" << std::endl;

  // PHY/MAC config
  Config::SetDefault ("ns3::MmWaveSidelinkMac::UseAmc", BooleanValue (true));
  Config::SetDefault ("ns3::MmWaveSidelinkMac::Mcs", UintegerValue (28));
  Config::SetDefault ("ns3::MmWavePhyMacCommon::CenterFreq", DoubleValue (frequency));
  Config::SetDefault ("ns3::MmWaveVehicularHelper::Bandwidth", DoubleValue (bwHz));
  Config::SetDefault ("ns3::MmWaveVehicularHelper::Numerology", UintegerValue (numerology));
  Config::SetDefault ("ns3::MmWaveVehicularHelper::ChannelModelType", StringValue ("V2V-Urban"));
  // 100 ms channel update; vehicles move about 2 m per update at 20 m/s.
  Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue (MilliSeconds (100)));
  Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod", TimeValue (MilliSeconds (100)));
  Config::SetDefault ("ns3::MmWaveVehicularNetDevice::RlcType", StringValue ("LteRlcUm"));
  Config::SetDefault ("ns3::MmWaveVehicularHelper::SchedulingPatternOption",
                      EnumValue (MmWaveVehicularHelper::OPTIMIZED));
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (500 * 1024));

  // nodes + mobility
  NodeContainer n;
  n.Create (numVehicles);
  g_nodes = n;                                 // for distance/position lookups in logs
  if (mobilityFile.empty ())
    {
      // synthetic convoy: increasing inter-vehicle gaps -> distance spread ~15..290 m
      MobilityHelper mob;
      mob.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
      mob.Install (n);
      for (uint32_t i = 0; i < numVehicles; ++i)
        {
          double y = ConvoyY (i);
          n.Get (i)->GetObject<MobilityModel> ()->SetPosition (Vector (0, y, 1.5));
          n.Get (i)->GetObject<ConstantVelocityMobilityModel> ()->SetVelocity (Vector (0, speed, 0));
        }
    }
  else
    {
      Ns2MobilityHelper ns2 (mobilityFile);
      ns2.Install ();   // expects the trace to define exactly numVehicles nodes
    }

  // devices
  Ptr<MmWaveVehicularHelper> helper = CreateObject<MmWaveVehicularHelper> ();
  helper->SetNumerology (numerology);
  NetDeviceContainer devs = helper->InstallMmWaveVehicularNetDevices (n);

  InternetStackHelper internet;
  internet.Install (n);
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ifaces = ipv4.Assign (devs);
  for (uint32_t i = 0; i < numVehicles; ++i)
    {
      uint32_t octet = ifaces.GetAddress (i).Get () & 0xFF;
      g_octetToNode[octet] = i;
    }

  BuildingsHelper::Install (n);
  helper->PairDevices (devs);

  // inject rain onto the shared native channel (no MilliCar source edits)
  if (trueRainRate > 0.0)
    {
      Ptr<MmWaveVehicularNetDevice> d0 = DynamicCast<MmWaveVehicularNetDevice> (devs.Get (0));
      Ptr<SpectrumChannel> channel = d0->GetPhy ()->GetSpectrumPhy ()->GetSpectrumChannel ();
      Ptr<WeatherRainLossModel> rain = CreateObject<WeatherRainLossModel> ();
      rain->SetBandGHz (bandGHz);   // the field itself lives in g_rain
      channel->AddPropagationLossModel (rain);
    }

  // logs: per-packet, per-rx-TB SINR/MCS, and a one-row meta
  if (logPkts)
    {
      g_pkt.open (outputDir + "pkts-" + simTag + ".csv");
      g_pkt << "txrx,time,src,dst,seq,bytes,latency_ms,dist_m,x,y\n";

      g_phy.open (outputDir + "phy-" + simTag + ".csv");
      g_phy << "time,rxnode,txnode,dist_m,sinr_db,numsym,tbsize,mcs\n";
      for (uint32_t i = 0; i < numVehicles; ++i)
        {
          Ptr<MmWaveVehicularNetDevice> di = DynamicCast<MmWaveVehicularNetDevice> (devs.Get (i));
          di->GetPhy ()->GetSpectrumPhy ()->SetSidelinkSinrReportCallback (
            MakeBoundCallback (&LogSinr, i));
        }

      // Meta carries the full rain-field description, so R(x,y,t) can be
      // reconstructed in post-processing.
      std::ofstream meta (outputDir + "meta-" + simTag + ".csv");
      meta << "regime,scenario,rain,band_ghz,bw_mhz,numveh,simtime,"
              "rainfield,cell_factor,rho0_km,peak_mmhr,cx0,cy0,vx,vy,path_samples\n"
           << regime << ',' << scenario << ',' << trueRainRate << ',' << bandGHz << ','
           << bwHz / 1e6 << ',' << numVehicles << ',' << simTime << ','
           << rainField << ',' << cellSizeFactor << ',' << g_rain.rho0Km << ','
           << g_rain.peakRate << ',' << g_rain.cx0 << ',' << g_rain.cy0 << ','
           << g_rain.vx << ',' << g_rain.vy << ',' << g_pathSamples << '\n';
      meta.close ();
    }

  // full-mesh CAM exchange (OnOff constant-rate; has a Tx trace)
  // one CAM (camBytes) every camIntervalMs -> DataRate = camBytes*8 / interval.
  DataRate camRate (static_cast<uint64_t> (camBytes * 8 * 1000.0 / camIntervalMs));
  uint16_t basePort = 5000;
  double appStart = 0.2, appStop = simTime;

  for (uint32_t j = 0; j < numVehicles; ++j)   // receiver j
    {
      uint16_t port = basePort + j;
      PacketSinkHelper sink ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), port));
      sink.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
      ApplicationContainer sinkApp = sink.Install (n.Get (j));
      sinkApp.Start (Seconds (0.0));
      if (logPkts)
        sinkApp.Get (0)->TraceConnectWithoutContext (
          "RxWithSeqTsSize", MakeBoundCallback (&LogRx, j));

      for (uint32_t i = 0; i < numVehicles; ++i)   // sender i
        {
          if (i == j) continue;
          OnOffHelper onoff ("ns3::UdpSocketFactory", InetSocketAddress (ifaces.GetAddress (j), port));
          onoff.SetConstantRate (camRate, camBytes);   // always on, one packet per interval
          onoff.SetAttribute ("EnableSeqTsSizeHeader", BooleanValue (true));
          ApplicationContainer c = onoff.Install (n.Get (i));
          c.Start (Seconds (appStart));
          c.Stop (Seconds (appStop));
          if (logPkts)
            c.Get (0)->TraceConnectWithoutContext (
              "TxWithSeqTsSize", MakeBoundCallback (&LogTx, i, j));
        }
    }

  Simulator::Stop (Seconds (simTime + 0.5));
  Simulator::Run ();
  Simulator::Destroy ();
  if (g_pkt.is_open ()) g_pkt.close ();
  if (g_phy.is_open ()) g_phy.close ();
  std::cout << "done " << simTag << std::endl;
  return 0;
}
