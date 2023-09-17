from multiprocessing import Value, Array


class Queue:
    def __init__(self, name, size):
        self.name = name
        self.size = Value("i", size, lock=False)
        self.rdPointer = Value("i", -1, lock=False) # refers to last read position
        self.wrPointer = Value("i", -1, lock=False) # refers to last written data item
        self.array = Array("f", size, lock=False)
        self.nrEvents = Value("i", 0, lock=False)
        self.nrSpikes = Value("i", 0, lock=False)
        self.parentName = "Aghaa"

    def setParentName(self, parentName):
        self.parentName = parentName

    def setLocation(self, location):
        self.location = location

    def inc(self, item):        
        return (item + 1) % (self.size.value)

    def incRdPointer(self):
        self.rdPointer.value = (self.rdPointer.value + 1) % (self.size.value)
        return self.rdPointer.value

    def incWrPointer(self):
        self.wrPointer.value = (self.wrPointer.value + 1) % (self.size.value)
        return self.wrPointer.value

    def isEmpty(self):
        return self.rdPointer.value ==self.wrPointer.value
    
    def isNotEmpty(self):
        return not(self.rdPointer.value ==self.wrPointer.value)

    def occupancy(self):
        if (self.rdPointer.value <= self.wrPointer.value):
            return self.wrPointer.value - self.rdPointer.value
        
        return self.size.value - (self.rdPointer.value - self.wrPointer.value)
    
    def isFull(self):
        return (self.rdPointer.value+1)%self.size.value == self.wrPointer.value

    def get(self, remove = True):
        if(self.isEmpty()):
            raise Exception("%s (son of %s): Queue is underflowed: trying to read an empty queue!"%(self.name, self.parentName))

        if(remove):
            nrSpikes = int(self.array[self.incRdPointer()])
            timestamp = self.array[self.incRdPointer()]
            srcLayer = int(self.array[self.incRdPointer()])
            height = int(self.array[self.incRdPointer()])
            width = int(self.array[self.incRdPointer()])
            resList = [timestamp, srcLayer, [height, width]]

            nrReadSpikes = 0
            while nrReadSpikes < nrSpikes:
                channel = int(self.array[self.incRdPointer()])
                value = self.array[self.incRdPointer()]
                nrReadSpikes += 1
                resList.append([channel, value])
            
            self.nrSpikes.value -= nrSpikes
            self.nrEvents.value -= 1

        else:
            rdPointer = self.inc(self.rdPointer.value)
            nrSpikes = int(self.array[rdPointer])
            
            rdPointer = self.inc(rdPointer)
            timestamp = int(self.array[rdPointer])
                        
            rdPointer = self.inc(rdPointer)
            srcLayer = int(self.array[rdPointer])
            
            rdPointer = self.inc(rdPointer)
            height = int(self.array[rdPointer])
            
            rdPointer = self.inc(rdPointer)
            width = int(self.array[rdPointer])
            resList = [timestamp, srcLayer, [height, width]]

            nrReadSpikes = 0
            while nrReadSpikes < nrSpikes:
                rdPointer = self.inc(rdPointer)
                channel = int(self.array[rdPointer])

                rdPointer = self.inc(rdPointer)
                value = self.array[rdPointer]
                nrReadSpikes += 1
                resList.append([channel, value])

        # from flatted list to event_list
        return resList
        
    def put(self, event):
        nrSpikes = len(event)-3
        eventSize = 2 * (nrSpikes) + 1 + 1 + 2 + 1
        
        if(eventSize + self.occupancy() > self.size.value):
            raise Exception("%s (son of %s): Queue is overflowed !" %(self.name, self.parentName))

        self.array[self.incWrPointer()] = nrSpikes
        self.array[self.incWrPointer()] = event[0]
        self.array[self.incWrPointer()] = event[1]
        self.array[self.incWrPointer()] = event[2][0]
        self.array[self.incWrPointer()] = event[2][1]

        for element in event[3:]:
            self.array[self.incWrPointer()] = element[0]
            self.array[self.incWrPointer()] = element[1]

        self.nrEvents.value += 1
        self.nrSpikes.value += nrSpikes

        return True

    def flush(self):
        while(self.isNotEmpty()):
            print("%s,%s"%(self.parentName, self.name),self.get())
    
    def printYourName(self):
        print("%s son of %s"%(self.name, self.parentName))
        print("occ=%d nrEv=%d nrSpike=%d"%(self.occupancy(), self.nrEvents.value, self.nrSpikes.value))
        print("rdpointer%d wrpointer=%d"%(self.rdPointer.value, self.wrPointer.value))
