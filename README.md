# Rollease Acmeda ARC Protocol Details
## Background
I bought a bunch of blinds from TheShadeStore with the idea of integrating them
into my home automation system. These are white-label blinds manufactured by
Rollease Acmeda.

TheShadeStore sold me handheld remotes [MT-0101-072002-A](https://www.automateshades.com/resource/product-quick-reference-guide-paradigm-plus-remote/) for each room which work reasonably well.

They also sold me an [Automate Pulse 2 Hub](https://rolleaseacmedamotors.com/products/rollease-acmeda-automate-pulse-2). The way this works is
that a phone app sends a command to the hub via wifi and
the hub issues a command to the blinds via the ARC protocol.

The hub does not work well in my experience. The installers from TheShadeStore were able to pair the hub to the blinds but
it would not stay connected.

I reached out to tech support on multiple occasions; they would send someone out to the house. They would spend time
moving things around and getting stuff paired again and
then it would stop working a day or two later.

The issue seems deeper than a radio strenth problem. When placing the blind right next to the hub (well within range of the handheld remote), the hub would not pair.

## Hardware
![Photo of PCB](data/Pluse-Hub2-rev1.6.jpg)
* U1: [ESP32-DOWD V1](https://documentation.espressif.com/esp32_datasheet_en.pdf) - MPU/WiFi
* U2: [Winbond 25Q32JV](https://www.winbond.com/hq/product/code-storage-flash-memory/serial-nor-flash/?__locale=en&partNo=W25Q32JV) - 32Mb NOR Flash
* U3: Unpopulated (TSSOP-8 or MSOP-8 package)
* U4: **Not found?**
* U5: [SMSC 8720A](https://ww1.microchip.com/downloads/en/DeviceDoc/8720a.pdf) - 0/100 Mbps Ethernet PHY transceiver
* U6: **Not found?**
* U7: [ST L051K86](https://www.st.com/en/microcontrollers-microprocessors/stm32l051k8.html) - MPU
* U8: [Si 44602A](https://www.silabs.com/documents/public/data-sheets/Si4463-61-60-C.pdf) - Radio tranciever
* U9: Unpopulated (SOIC-8, TSSOP-8, or MSOP-8)
* U10: **Not found?**
* U11: **Unknown** - silver near ISM antenna
* U12: **Unknown** - near WiFi antenna
* U13: Unpopulated (DFN-8 or MSOP-8)
* U14: **Unknown** - small black bga; plausibly `ATECC508A` for AWS IoT authentication
* U15: **Unknown** - marked with 8536; near multi-color LED.
* U16: Unpopulated (SOT-23-6)
* U17: **Unknown** - LDO?

There are a lot of test points; the largest designator I see is TP35 but I only count 32. TP13-16 break out the SPI pins on the si4460.

### Test Points for Si4460 SPI
In the orientation of the photo, the SPI pins are broken out as test points.
They are left and below the chip.
|Pin|TP  |Description
|---|----|-----------
|12 |TP15|SCLK
|15 |TP16|nSEL
|13 |TP14|SDO
|14 |TP13|SDI

### ESP32 flashing pins
|Pin|TP        |Description
|---|----------|-----------
|   |       TP1|3V3 (also unpopulated header J3)
|   |       TP4|EN (also unpopulated header J3)
|   |       TP5|Ground (also unpopulated header J3)
|   |       TP6|Ground (also unpopulated header J2)
|   |       TP1|Ground (also unpopulated header J4)
|   |       TP8|RXI (also unpopulated header J2)
|   |(unlabled)|TXD (also unpopulated header J2)
|   |       "0"|GPIO0 (no header, above eth port on right)

## SPI Trace
I was able to follow the directions to flash a hub with ESPHome; while I was
soldering to the PCB, I attached wires to the SPI and used that to capture the
SPI traffic between the 6640 and the STM32L051 chip. See [PulseView file](data/SPI%20Boot%20Capture.sr).

There were some malformed updates, but here are the distinct `SET_PROPERTY` commands (starting
with `0x11`, then a group, then a count of bytes to set, then a starting offset, then count bytes):
```
11 01 02 00 05 3A
11 02 01 00 0A
11 10 01 04 31
11 11 05 00 03 B4 2B B4 2B
11 12 01 10 08
11 12 03 6F FF FF FF
11 12 04 36 FF FF FF FF
11 12 07 0E 01 04 80 00 3F 00 2A
11 12 07 22 01 04 80 00 3F 00 0A
11 12 0A 00 04 01 08 FF FF 20 82 00 2A 01
11 20 02 0B 05 76
11 20 02 50 84 0A
11 20 02 54 83 41
11 20 03 30 02 10 80
11 20 06 5A 89 62 64 06 78 24
11 20 07 00 03 00 07 0C 35 00 09
11 20 08 38 11 15 15 80 1A 40 00 00
11 20 09 45 8F 00 DE 01 00 44 06 0A 1A
11 20 0C 18 01 80 08 03 80 00 20 20 00 E8 00 5E
11 20 0C 24 05 76 1A 02 B9 02 C0 00 94 23 81 56
11 21 06 2E 7D 40 A0 44 28 20 57
11 21 0C 00 CC A1 30 A0 21 D1 B9 C9 EA 05 12 11
11 21 0C 0C 0A 04 15 FC 03 00 CC A1 30 A0 21 D1
11 21 0C 18 B9 C9 EA 05 12 11 0A 04 15 FC 03 00
11 22 04 00 18 10 C0 1D
11 40 08 00 38 0E DA 74 44 44 20 FE
11 61 0C 18 B9 C9 EA 05 12 11 0A 04 15 FC 03 00
```

That maps to:
| Property | Name | Meaning | Value |
|---------|------|---------|--------|
| 01:00 | INT_CTL_ENABLE | Master interrupt enables |05|
| 01:01 | INT_CTL_PH_ENABLE | Packet handler interrupt enables |3A|
| 02:00 | FRR_CTL_A_MODE ||0A|
| 10:04 | MODEM_MOD_TYPE || 31 = GFSK |
| 11:00 | MODEM_CHFLT_RX1_CHFLT_COE13_7_0 ||03|
| 11:01 | MODEM_CHFLT_RX1_CHFLT_COE12_7_0 ||B4|
| 11:02 | MODEM_CHFLT_RX1_CHFLT_COE11_7_0 ||2B|
| 11:03 | MODEM_CHFLT_RX1_CHFLT_COE10_7_0 ||B4|
| 11:04 | MODEM_CHFLT_RX1_CHFLT_COE9_7_0 ||2B|
| 12:00 | PA_MODE ||04|
| 12:01 | PA_PWR_LVL ||01|
| 12:02 | PA_BIAS_CLKDUTY ||08|
| 12:03 | PA_TC ||FF|
| 12:04 | PA_RAMP_EX ||FF|
| 12:05 | PA_RAMP_DOWN_DELAY ||20|
| 12:06 | PA_DIG_PWR_SEQ_CONFIG ||82|
| 12:07 | PA_DIG_PWR_SEQ_DELAY ||00|
| 12:08 | PA_DIG_PWR_SEQ_STEP ||2A|
| 12:09 | PA_DIG_PWR_SEQ_STEP2 ||01|
| 12:0E–12:39|undocumented PA/GPIO/Sequencer internal||01,04,08,80,00,3F,00,2A,01,04,80,00,3F,00,0A,FF,FF,FF,FF|
| 12:6F–12:71|undocumented PA/GPIO/Sequencer internals||FF,FF,FF|
| 20:00 | SYNTH_PFDCP_CPFF ||03|
| 20:01 | SYNTH_PFDCP_CPINT ||00|
| 20:02 | SYNTH_PFDCP_CPINC ||07|
| 20:03 | SYNTH_PFDCP_CPI ||0C|
| 20:04 | SYNTH_PFDCP_FDCP ||35|
| 20:05 | SYNTH_LPFILT3 ||00|
| 20:06 | SYNTH_LPFILT2 ||09|
| 20:0B | SYNTH_LPFILT1 ||05|
| 20:0C | SYNTH_LPFILT0 ||76|
| 20:18–20:1F | SYNTH_VCO_KV / VCO calibration ||01,80,08,03,80,00,20,20|
| 20:20–20:23 | SYNTH_VCO_KVCAL ||00,E8,00,5E|
| 20:24–20:2F | SYNTH_VCO parameters ||05,76,1A,02,B9,02,C0,00,94,23,81,56|
| 20:30 | SYNTH_LPFILT3_RX ||02|
| 20:31 | SYNTH_LPFILT2_RX ||10|
| 20:32 | SYNTH_LPFILT1_RX ||80|
| 20:38–20:3F | SYNTH_LPFILT_RX path ||11,15,15,80,1A,40,00,00|
| 20:45–20:4D | SYNTH_MISC / VCO / charge pump ||8F,00,DE,01,00,44,06,0A,1A|
| 20:50–20:55 | SYNTH_DIVIDER / BAND / OUTDIV ||84,0A,83,41|
| 20:5A–20:5F | SYNTH calibration / VCO bias ||89,62,64,06,78,24|
| 21:00–21:23 | MATCH_VALUE_x (documented as MATCH registers) ||CC,A1,30,A0,21,D1,B9,C9,EA,05,12,11,0A,04,15,FC,03,00,CC,A1,30,A0,21,D1,B9,C9,EA,05,12,11,0A,04,15,FC,03,00|
| 21:2E–21:33 | MATCH_VALUE_x ||7D,40,A0,44,28,20|
| 22:00 | FREQ_CONTROL_INTE ||18|
| 22:01 | FREQ_CONTROL_FRAC_2 ||10|
| 22:02 | FREQ_CONTROL_FRAC_1 ||C0|
| 22:03 | FREQ_CONTROL_FRAC_0 ||1D|
| 40:00–40:07 | Undocumented MODEM internal parameters ||38, 0E, DA, 74, 44, 44, 20, FE|
| 61:18–61:23 | Undocumented AGC/timing/slicer parameters ||B9,C9,EA,05,12,11,0A,04,15,FC,03,00|

## Documents
[FCC Details](https://fccid.io/pdf.php?id=4975342). The 433MHz grant is specifically for 433.92MHz-433.92MHz (no range)

## Other work
There seems to be a high level protocol `'!' + address[0] + address[1] + address[3] + command + param[0] + param[1] ... + ';'`.
|command|data|description
|-|-|-
|o||Open/Up
|c||Close/Down
|s||Stop
|oA||Jog Open/Up
|cA||Jog Close/Down
|m|DDD|Move by percentage
|b|DDD|Rotate angle by percentage

### HACS component
I'm using [Home Assistant HACS Component](https://github.com/sillyfrog/Automate-Pulse-v2?tab=readme-ov-file) and [Python Module](https://github.com/sillyfrog/aiopulse2). This would be my preferred option except that the hub itself has problems communicating with the blinds.

### ESPHome component
I just found an [ESPHome component](https://github.com/redstorm1/arc-bridge) (with [blog post](https://www.geektech.co.nz/esphome-pulse-2-hub)). It has this helpful snippet:
> #### Electrical Architecture
> * The ESP32 communicates with the STM32L051 over UART (115200 baud, 8N1).
> * The STM32 controls the RF transmitter for ARC radio communication.
> * The LAN8720A provides wired Ethernet; clocked externally through GPIO 0.
> * The PCA9554 handles LED outputs and expansion pins via I²C (SDA = GPIO 14, SCL = GPIO 4).
> * Power distribution is 3.3 V throughout the logic section.
>
> This architecture allows ESPHome to use Ethernet networking while simultaneously driving the RF microcontroller through UART.

I suspect, based on PCB layout, the `PC9554` is `U15` (labeled `8536`). This is not the most important part.

### Serial Protocol
I found a documents with the [Serial Protocol for Pulse Blinds](https://www.avoutlet.com/images/product/additional/r/pulse-serial-instructions.pdf) on this
[Bond Home forum thread](https://forum.bondhome.io/t/rollease-acmeda-and-dooya-motorized-shades/413/3). It suggests 9600 baud, 8N1.

## Capture Process
### GQRX Survey
Here's a screen shot from GQRX:
![GQRX Screen Shot](data/GQRX_f433000000_s2000000_a0_l16_g2.png)
It shows energy around 433.92MHz (as expected). But the DC spike seems to be at
433.277MHz rather than the 433.0MHz targeted by the tuning parameter.

### `hackrf_transfer` capture
Here's my attempt to replicate that with the command line tool.
```bash
hackrf_transfer -f 433125000 -s 20000000 -a 0 -l 16 -g 2 -n 80000000 -r $datadir/remote_dr2_f433125000_s2000000_a0_l16_g2.iq
```
Which yields:
```
call hackrf_set_sample_rate(20000000 Hz/20.000 MHz)
call hackrf_set_hw_sync_mode(0)
call hackrf_set_freq(433125000 Hz/433.125 MHz)
call hackrf_set_amp_enable(0)
samples_to_xfer 80000000/80Mio
Stop with Ctrl-C
39.6 MiB / 1.014 sec = 39.0 MiB/second, average power -33.2 dBfs
39.3 MiB / 0.996 sec = 39.5 MiB/second, average power -33.1 dBfs
39.6 MiB / 1.001 sec = 39.6 MiB/second, average power -33.1 dBfs
39.6 MiB / 1.002 sec = 39.5 MiB/second, average power -33.1 dBfs
 2.1 MiB / 0.047 sec = 44.3 MiB/second, average power -33.3 dBfs

Exiting...
Total time: 4.05986 s
hackrf_stop_rx() done
hackrf_close() done
hackrf_exit() done
fclose() done
exit
```

I then convert the int8 to complex float 32:
```python
import numpy as np

def load_hackrf_iq(path):
    raw = np.fromfile(path, dtype=np.int8)
    iq = raw.reshape(-1, 2)
    return (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / 128.0

def save_to_gqrx_float32(x, path):
    interleaved = np.empty(2 * len(x), dtype=np.float32)
    interleaved[0::2] = x.real.astype(np.float32)
    interleaved[1::2] = x.imag.astype(np.float32)
    interleaved.tofile(path)

samples = load_hackrf_iq('data/remote_dr2_f433125000_s2000000_a0_l16_g2.iq')
save_to_gqrx_float32(samples, 'generated/remote_dr2_f433125000_s2000000_a0_l16_g2.c32')
```

After opening GQRX with the device string of `file=generated/dining_room_2_f433000000_s2000000.c32,freq=433.0e6,rate=20e6,repeat=true,throttle=true`, and setting the frequency to `0` in the **Receiver Options** tab, I get this:
![GQRX Screen Shot](data/GQRX_f433125000_s2000000_a0_l16_g2.png)