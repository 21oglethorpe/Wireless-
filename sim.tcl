#configurations for the node
set val(chan)		Channel/WirelessChannel		;
set val(prop)		Propagation/TwoRayGround	;
set val(netif)		Phy/WirelessPhy			;
set val(mac)		Mac/802_11				;
set val(ifq)		Queue/DropTail/PriQueue		;
set val(ll)			LL					;
set val(ant)		Antenna/OmniAntenna		;
set val(ifqlen)		50					;
set val(nn)			6					;
set val(rp)			AODV					;
set val(x)			500					;
set val(y)			500					;

#Create a simulator object
set ns [new Simulator]

set tracefile [open out.tr w]
$ns trace-all $tracefile

#Open the nam trace file
set nf [open out.nam w]
$ns namtrace-all-wireless $nf $val(x) $val(y)

#I think this is the map lol
set topo [new Topography]
$topo load_flatgrid $val(x) $val(y)
#This is heresy to Christianity 
create-god $val(nn)

#You can create multiple channels but IDK how to set up multiple channels 
#cuz it seems to only run one at a time when i try to set up multiple. Maybe 
#because I only tried changing it at the same node? idk
set channel1 [new $val(chan)]
set channel2 [new $val(chan)]
set channel3 [new $val(chan)]
$ns node-config -adhocRouting $val(rp) \
		    -llType $val(ll) \
		    -macType $val(mac) \
		    -ifqType $val(ifq) \
		    -ifqLen $val(ifqlen) \
		    -antType $val(ant) \
		    -propType $val(prop) \
		    -phyType $val(netif) \
		    -topoInstance $topo \
		    -agentTrace ON \
		    -macTrace ON \
		    -routerTrace ON \
		    -movementTrace ON \
		    -channel $channel1


#Create nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]

#Set up size of nodes for simulation
$ns initial_node_pos $n2 20
$ns initial_node_pos $n0 20
$ns initial_node_pos $n1 20
$ns initial_node_pos $n3 20

#Create a duplex link between the nodes
$ns duplex-link $n0 $n1 1Mb 10ms DropTail
$ns duplex-link $n0 $n2 1Mb 10ms DropTail
$ns duplex-link $n0 $n3 1Mb 10ms DropTail

#Create a TCP agent and attach it to node n0
set udp0 [new Agent/TCP]
set udp1 [new Agent/TCP]
set udp2 [new Agent/TCP]



$ns attach-agent $n0 $udp0
$ns attach-agent $n0 $udp1
$ns attach-agent $n0 $udp2


# Create a FTP traffic source and attach it to udp0
set cbr0 [new Application/FTP]
set cbr1 [new Application/FTP]
set cbr2 [new Application/FTP]

#I can manually set size and speed of packets for each connection
$cbr0 set packetSize_ 2000
$cbr0 set interval_ 0.05
$cbr0 attach-agent $udp0

$cbr1 set packetSize_ 2000
$cbr1 set interval_ 0.05
$cbr1 attach-agent $udp1
$cbr2 set packetSize_ 2000
$cbr2 set interval_ 0.05
$cbr2 attach-agent $udp2

#Create a Null agent (a traffic sink) and attach it to node n1
set null0 [new Agent/TCPSink]
set null1 [new Agent/TCPSink]
set null2 [new Agent/TCPSink]

$ns attach-agent $n1 $null0
$ns attach-agent $n2 $null1
$ns attach-agent $n3 $null2

#Connect the traffic source with the traffic sink
$ns connect $udp0 $null0  
$ns connect $udp1 $null1 
$ns connect $udp2 $null2  

#Set initial positions
$n0 set X_ 300.0
$n0 set Y_ 200.0
$n0 set Z_ 0.0

$n1 set X_ 240.0
$n1 set Y_ 280.0
$n1 set Z_ 0.0

$n2 set X_ 300.0
$n2 set Y_ 300.0
$n2 set Z_ 0.0

$n3 set X_ 360.0
$n3 set Y_ 280.0
$n3 set Z_ 0.0
#Schedule events for the CBR agent
#I think setdest lets nodes move but i don't think just removing them
#works with how the simulator works but I also never tried lol. The 
#destinations are the same as initial so they don't move.
$ns at 0.1 "$n0 setdest 300.0 200.0 0.0"
$ns at 0.1 "$n1 setdest 240.0 220.0 0.0"
$ns at 0.1 "$n2 setdest 300.0 300.0 0.0"
$ns at 0.1 "$n3 setdest 360.0 280.0 0.0"
$ns at 0.1 "$cbr0 start"
$ns at 0.1 "$cbr1 start"
$ns at 0.1 "$cbr2 start"

#Define a 'finish' procedure to close out simulator
proc finish {} {
        global ns tracefile nf
        $ns flush-trace
	#Close the trace file
      close $tracefile  
	close $nf
	#Execute nam on the trace file
        #exec nam out.nam &
        exit 0
}
#Call the finish procedure after 5 seconds of simulation time
$ns at 5.0 "finish"


#Run the simulation
$ns run
