from xml.etree.ElementTree import parse, Element
import argparse


def main():
    parser = argparse.ArgumentParser(description='edit ipi input xml')
    parser.add_argument('-i', '--in', dest='filename_in', type=str, help='input file name', required=True)
    parser.add_argument('-a', '--add', dest='address', type=str, help='address')
    parser.add_argument('-n', '--nbeads', dest='nbeads', type=str, help='number of beads')
    parser.add_argument('-t', '--temp', dest='temp', type=str, help='temperature')
    parser.add_argument('-p', '--pres', dest='pressure', type=str, help='pressure')
    parser.add_argument('-v', '--velocity', dest='velocity', type=str, help='velocity')
    parser.add_argument('-o', '--out', dest='filename_out', type=str, help='input file name', required=True)
    args = parser.parse_args()
    doc = parse(args.filename_in)
    if args.nbeads:
        initialize = doc.find('system/initialize')
        # print(initialize.get('nbeads'))
        initialize.set('nbeads', args.nbeads)
        # print(initialize.get('nbeads'))
    if args.velocity:
        velocities = doc.find('system/initialize/velocities')
        # print(velocities.text)
        velocities.text = args.velocity
        # print(velocities.text)
    if args.temp:
        temperature = doc.find('system/ensemble/temperature')
        # print(temperature.text)
        temperature.text = args.temp
        # print(temperature.text)
    if args.pressure:
        pressure = doc.find('system/ensemble/pressure')
        # print(pressure.text)
        pressure.text = str(float(args.pressure)*1000.0)
        # print(pressure.text)
    if args.address:
        address = doc.find('ffsocket/address')
        address.text = args.address
    doc.write(args.filename_out)


if __name__ == '__main__':
    main()
