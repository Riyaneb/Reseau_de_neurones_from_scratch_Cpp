#ifndef RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
#define RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
#include <cassert>
#include "math/Types.hpp"

class Shape {
public:
    Shape(Index c, Index h ,Index w) {
        assert((c > 0 && w > 0 && h > 0) && "Erreur : Dimension de la structure shape incorrecte");
        channel = c;
        height = h;
        width = w;
    }
    Index total_values() const;
    Index get_column(Index c, Index h, Index w) const;

    inline Index get_channel() const {
        return channel;
    }

    inline Index get_height() const {
        return height;
    }

    inline Index get_width() const {
        return width;
    }
private:
    Index channel;
    Index height;
    Index width;
};

#endif //RESEAUDENEURONEFROMSCRATCH_SHAPE_HPP
