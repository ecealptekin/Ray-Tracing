#pragma once

#include <iostream>
#include <fstream>

#include "PixelBuffer.h"

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

namespace IO
{
    void write_as_PPM(const PixelBuffer& pixel_buffer, std::ostream& output, int samples_per_pixel)
    {
        output << "P3\n" << pixel_buffer.dimensions.x << ' ' << pixel_buffer.dimensions.y << "\n255\n";

        const int total_pixels = pixel_buffer.dimensions.x * pixel_buffer.dimensions.y;
        for (int i = 0; i < total_pixels; ++i)
        {
          
            auto v = pixel_buffer.get(i) * 255.99;
            
            auto r = v.x;
            auto g = v.y;
            auto b = v.z;
            
            // Divide the color by the number of samples and gamma-correct for gamma=2.0.
            auto scale = 1.0 / samples_per_pixel;
            r = sqrt(scale * r);
            g = sqrt(scale * g);
            b = sqrt(scale * b);
            
            int ir = static_cast<int>(256 * clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b, 0.0, 0.999));

            output << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}

/*
#pragma once

#include <iostream>
#include <fstream>

#include "PixelBuffer.h"

namespace IO
{
    void write_as_PPM(const PixelBuffer& pixel_buffer, std::ostream& output)
    {
        output << "P3\n" << pixel_buffer.dimensions.x << ' ' << pixel_buffer.dimensions.y << "\n255\n";

        const int total_pixels = pixel_buffer.dimensions.x * pixel_buffer.dimensions.y;
        for (int i = 0; i < total_pixels; ++i)
        {
            auto v = pixel_buffer.get(i) * 255.99;

            int ir = static_cast<int>(v.r);
            int ig = static_cast<int>(v.g);
            int ib = static_cast<int>(v.b);

            output << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}
*/
