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
* U6: Not found?
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

## Equipment
### RTL-SDR
### HackRF1
### Yardstick1