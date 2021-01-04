from __future__ import division


class PenaltyBuilder(object):
    def __init__(self, length_pen):
        self.length_pen = length_pen

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    def length_wu(self, beam, logprobs, alpha=0.0):
        modifier = ((5 + len(beam.next_ys)) ** alpha) / ((5 + 1) ** alpha)
        return logprobs / modifier

    def length_average(self, beam, logprobs, alpha=0.0):
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0.0, beta=0.0):
        return logprobs
