#include <iostream>
#include <vector>
#include "glm/glm.hpp"

#include "IO.h"
#include "PixelBuffer.h"

#include <memory>

const double infinity = std::numeric_limits<double>::infinity();

struct Ray
{
    glm::dvec3 origin, direction;

    Ray() {}
    Ray(const glm::dvec3& origin, const glm::dvec3& direction)
        : origin(origin), direction(direction)
    {}

    glm::dvec3 at(double t) const
    {
        return origin + t * direction;
    }
};

struct hit_record;

class material {
    public:
        virtual glm::dvec3 emitted(double u, double v, const glm::dvec3& p) const {
        return glm::dvec3(0,0,0);
        }
        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const = 0;
};

struct hit_record
{
    glm::dvec3 p;   //point3
    glm::dvec3 normal;
    std::shared_ptr<material> mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;

    inline void set_face_normal(const Ray& r, const glm::dvec3& outward_normal)
    {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
    
};


struct Camera
{
    glm::dvec3 position, left_bottom, horizontal, vertical;
    double focal_length;

    Camera(const glm::dvec3& position, const glm::dvec3& target, const PixelBuffer& pixel_buffer)
        : position(position), focal_length(1)
    {
        glm::dvec3 up(0, 1, 0);
        auto forward = glm::normalize(target - position);
        auto right = glm::normalize(glm::cross(forward, up));
        up = glm::normalize(glm::cross(forward, -right));

        horizontal = right * (double(pixel_buffer.dimensions.x) / double(pixel_buffer.dimensions.y));
        vertical = up * 1.;
        left_bottom = position + forward * focal_length - horizontal * 0.5 - vertical * 0.5;
    }

    glm::dvec3 raster_to_world(const glm::dvec2& r)
    {
        //2D to 3D
        //W = C0 + rx * CH + ry * CV
        return left_bottom + r.x * horizontal + r.y * vertical;
    }
};




/*
class sphere : public hittable {
    public:
        sphere() {}
        sphere(glm::dvec3 cen, double r) : center(cen), radius(r) {};

        virtual bool hit(
            const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        glm::dvec3 center;
        double radius;
};

bool sphere::hit(const Ray& ray, double t_min, double t_max, hit_record& rec) const
{
   glm::dvec3 oc = ray.origin - center;
   //auto a = ray.direction.length_squared();
   auto a = (ray.direction.x * ray.direction.x) + (ray.direction.y * ray.direction.y) + (ray.direction.z * ray.direction.z);
   auto half_b = dot(oc, ray.direction);
   auto tr = (oc.x * oc.x) + (oc.y * oc.y) + (oc.z * oc.z);
   auto c = tr - radius*radius;
    
    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;  //t double
    rec.p = ray.at(rec.t);
    rec.normal = (rec.p - center) / radius;

    return true;
}
*/


using std::shared_ptr;
using std::make_shared;



double length_squared(glm::dvec3 vector3)
{
    return (vector3.x * vector3.x) + (vector3.y * vector3.y) + (vector3.z * vector3.z);
}

inline double random_double()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max-min) * random_double();
}

glm::dvec3 random_in_unit_sphere()
{
    while (true) {
        auto p = glm::dvec3(random_double(-1,1), random_double(-1,1), random_double(-1,1));
        if (length_squared(p) >= 1) continue;
        return p;
    }
}

//True Lambertian Reflection
glm::dvec3 random_unit_vector() {
    return glm::normalize(random_in_unit_sphere());
}


bool near_zero(glm::dvec3 vector)
{
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(vector.x) < s) && (fabs(vector.y) < s) && (fabs(vector.z) < s);
}

glm::dvec3 reflect(const glm::dvec3& v, const glm::dvec3& n)
{
    return v - 2 * dot(v,n) * n;
}

class metal : public material {
    public:
        metal(const glm::dvec3& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            glm::dvec3 reflected = reflect(glm::normalize(r_in.direction), rec.normal);
            scattered = Ray(rec.p, reflected + fuzz*random_in_unit_sphere());
            attenuation = albedo;
            return (dot(scattered.direction, rec.normal) > 0);
        }

    public:
        glm::dvec3 albedo;
        double fuzz;
};

glm::dvec3 refract1(const glm::dvec3& uv, const glm::dvec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    glm::dvec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    glm::dvec3 r_out_parallel = -sqrt(fabs(1.0 - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

class dielectric : public material {
    public:
        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            attenuation = glm::dvec3(1.0, 1.0, 1.0);
            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            glm::dvec3 unit_direction = glm::normalize(r_in.direction);
            
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            glm::dvec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
            {
              direction = reflect(unit_direction, rec.normal);
            }
            
            else
            {
              direction = refract1(unit_direction, rec.normal, refraction_ratio);
            }
              
            scattered = Ray(rec.p, direction);
            
            return true;
        }

    public:
        double ir; // Index of Refraction
    
    private:
           static double reflectance(double cosine, double ref_idx)
    {
           // Use Schlick's approximation for reflectance.
           auto r0 = (1-ref_idx) / (1+ref_idx);
           r0 = r0*r0;
           return r0 + (1-r0)*pow((1 - cosine),5);
    }
};

class texture {
    public:
        virtual glm::dvec3 value(double u, double v, const glm::dvec3& p) const = 0;
};

class solid_color : public texture {
    public:
        solid_color() {}
        solid_color(glm::dvec3 c) : color_value(c) {}

        solid_color(double red, double green, double blue)
          : solid_color(glm::dvec3(red,green,blue)) {}

        virtual glm::dvec3 value(double u, double v, const glm::dvec3& p) const override {
            return color_value;
        }

    private:
        glm::dvec3 color_value;
};

class diffuse_light : public material  {
    public:
        diffuse_light(shared_ptr<texture> a) : emit(a) {}
        diffuse_light(glm::dvec3 c) : emit(make_shared<solid_color>(c)) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            return false;
        }

        virtual glm::dvec3 emitted(double u, double v, const glm::dvec3& p) const override {
            return emit->value(u, v, p);
        }

    public:
        shared_ptr<texture> emit;
};

class aabb {
    public:
        aabb() {}
        aabb(const glm::dvec3& a, const glm::dvec3& b) { minimum = a; maximum = b;}

        glm::dvec3 min() const {return minimum; }
        glm::dvec3 max() const {return maximum; }

        bool hit(const Ray& r, double t_min, double t_max) const {
            for (int a = 0; a < 3; a++) {
                auto t0 = fmin((minimum[a] - r.origin[a]) / r.direction[a],
                               (maximum[a] - r.origin[a]) / r.direction[a]);
                auto t1 = fmax((minimum[a] - r.origin[a]) / r.direction[a],
                               (maximum[a] - r.origin[a]) / r.direction[a]);
                t_min = fmax(t0, t_min);
                t_max = fmin(t1, t_max);
                if (t_max <= t_min)
                    return false;
            }
            return true;
        }
    
        glm::dvec3 minimum;
        glm::dvec3 maximum;
};

class hittable
{
    public:
        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
        virtual bool bounding_box(double time0, double time1, aabb& output_box) const = 0;
};

class hittable_list : public hittable {
    
    public:
        hittable_list() {}
        hittable_list(shared_ptr<hittable> object) { add(object); }

        void clear() { objects.clear(); }
        void add(shared_ptr<hittable> object) { objects.push_back(object); }

        virtual bool hit(
            const Ray& r, double t_min, double t_max, hit_record& rec) const override;
    
        virtual bool bounding_box(
                   double time0, double time1, aabb& output_box) const override;

    public:
        std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

aabb surrounding_box(aabb box0, aabb box1)
{
    glm::dvec3 small(fmin(box0.min().x, box1.min().x),
                 fmin(box0.min().y, box1.min().y),
                 fmin(box0.min().z, box1.min().z));

    glm::dvec3 big(fmax(box0.max().x, box1.max().x),
               fmax(box0.max().y, box1.max().y),
               fmax(box0.max().z, box1.max().z));

    return aabb(small,big);
}

bool hittable_list::bounding_box(double time0, double time1, aabb& output_box) const {
    if (objects.empty()) return false;

    aabb temp_box;
    bool first_box = true;

    for (const auto& object : objects) {
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

class xy_rect : public hittable {
    public:
        xy_rect() {}

        xy_rect(double _x0, double _x1, double _y0, double _y1, double _k,
            shared_ptr<material> mat)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(glm::dvec3(x0,y0, k-0.0001), glm::dvec3(x1, y1, k+0.0001));
            return true;
        }

    public:
        shared_ptr<material> mp;
        double x0, x1, y0, y1, k;
};

bool xy_rect::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    auto t = (k-r.origin.z) / r.direction.z;
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin.x + t*r.direction.x;
    auto y = r.origin.y + t*r.direction.y;
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    auto outward_normal = glm::dvec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}

const double pi = 3.1415926535897932385;

class sphere : public hittable {
    
public:
    sphere() {}
    sphere(glm::dvec3 cen, double r, std::shared_ptr<material> m)
    : center(cen), radius(r), mat_ptr(m) {};

    virtual bool hit(
       const Ray& r, double t_min, double t_max, hit_record& rec) const override;
    
    virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
    
public:
    glm::dvec3 center;
    double radius;
    std::shared_ptr<material> mat_ptr;
    
    private:
           static void get_sphere_uv(const glm::dvec3& p, double& u, double& v) {
               // p: a given point on the sphere of radius one, centered at the origin.
               // u: returned value [0,1] of angle around the Y axis from X=-1.
               // v: returned value [0,1] of angle from Y=-1 to Y=+1.
               //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
               //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
               //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

               auto theta = acos(-p.y);
               auto phi = atan2(-p.z, p.x) + pi;

               u = phi / (2*pi);
               v = theta / pi;
           }

};
    
    bool sphere::hit(const Ray& ray, double t_min, double t_max, hit_record& rec) const
    {
        glm::dvec3 oc = ray.origin - center;
        
        //auto a = ray.direction.length_squared();
        auto a = (ray.direction.x * ray.direction.x) + (ray.direction.y * ray.direction.y) + (ray.direction.z * ray.direction.z);
        auto half_b = dot(oc, ray.direction);
        auto tr = (oc.x * oc.x) + (oc.y * oc.y) + (oc.z * oc.z);
        auto c = tr - radius*radius;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant < 0)
        {
            return false;
        }
        auto sqrtd = sqrt(discriminant);
        
        // Find the nearest root that lies in the acceptable range.
           auto root = (-half_b - sqrtd) / a;
           if (root < t_min || t_max < root) {
               root = (-half_b + sqrtd) / a;
               if (root < t_min || t_max < root)
                   return false;
           }

           rec.t = root;  //t double
           rec.p = ray.at(rec.t);
           rec.normal = (rec.p - center) / radius;
           glm::dvec3 outward_normal = (rec.p - center) / radius;
           rec.set_face_normal(ray, outward_normal);
           get_sphere_uv(outward_normal, rec.u, rec.v);
           rec.mat_ptr = mat_ptr;

        
           return true;
    }

bool sphere::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(
        center - glm::dvec3(radius, radius, radius),
        center + glm::dvec3(radius, radius, radius));
    return true;
}


glm::dvec3 ray_color(const Ray& r, const glm::dvec3& background, const hittable& world, int depth)
{
    hit_record rec;

    if (depth <= 0)
    {
        return glm::dvec3(0,0,0);
    }
    
    if(!world.hit(r, 0.001, infinity, rec)) //shadow acne
    {
        return background;
    }
        
        //update the ray_color() function to use the new random direction generator
        //glm::dvec3 target = rec.p + rec.normal + random_unit_vector();
        //return 0.5 * ray_color(Ray(rec.p, target - rec.p), world, depth-1);
        
        
    Ray scattered;
    glm::dvec3 attenuation;
    glm::dvec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        
    if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
    {
        return emitted;
    }
        
   return emitted + attenuation * ray_color(scattered, background, world, depth-1);
}

/*
glm::dvec3 ray_color(const Ray& r, const hittable& world, int depth)
{
    hit_record rec;

    if (depth <= 0)
    {
        return glm::dvec3(0,0,0);
    }
    
    if(world.hit(r, 0.001, infinity, rec)) //shadow acne
    {
        
        //update the ray_color() function to use the new random direction generator
        glm::dvec3 target = rec.p + rec.normal + random_unit_vector();
        return 0.5 * ray_color(Ray(rec.p, target - rec.p), world, depth-1);
        
    }
    
    glm::dvec3 unit = glm::normalize(r.direction);
    auto t = 0.5 * (unit.y + 1.0);
    return (1.0-t) * glm::dvec3(1.0, 1.0, 1.0) + t * glm::dvec3(0.0, 0.0, 0.0);
}
*/

class lambertian : public material {
    public:
        lambertian(const glm::dvec3& a) : albedo(make_shared<solid_color>(a)) {}
        lambertian(shared_ptr<texture> a) : albedo(a) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            auto scatter_direction = rec.normal + random_unit_vector();
            
            // Catch degenerate scatter direction
            if (near_zero(scatter_direction))
            {
                scatter_direction = rec.normal;
            }
            
            scattered = Ray(rec.p, scatter_direction);
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }

    public:
        shared_ptr<texture> albedo;
};


hittable_list simple_light()
{
    hittable_list objects;

    auto pertext = make_shared<metal>(glm::dvec3(0.3, 0.2, 0.1), 0.0);
    objects.add(make_shared<sphere>(glm::dvec3(1,0,0), 0.5, pertext));
    objects.add(make_shared<sphere>(glm::dvec3(0,2,0), 0.5, pertext));

    auto difflight = make_shared<diffuse_light>(glm::dvec3(7,7,7));
    objects.add(make_shared<xy_rect>(-1, 1, -3, 2, -2, difflight));
    objects.add(make_shared<xy_rect>(2, 4, -3, 2, -2, difflight));
    
    /*
    auto material_ground = make_shared<lambertian>(glm::dvec3(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(glm::dvec3(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.5);
    auto material_right = make_shared<metal>(glm::dvec3(0.8, 0.6, 0.2), 0.0);
    
    objects.add(make_shared<sphere>(glm::dvec3(0.0,-150.0,-1.0), 100.0, material_ground));
    objects.add(make_shared<sphere>(glm::dvec3(-2.5,0.0, 0.0), 0.5, material_center));
    objects.add(make_shared<sphere>(glm::dvec3(0.0, 0.0, 0.0), 2.0, material_left));
    objects.add(make_shared<sphere>(glm::dvec3(2.5, 0.0, 0.0), 0.5, material_right));
    */
    
    /*
    auto material_ground = make_shared<metal>(glm::dvec3(0.8, 0.6, 0.2), 0.5);
    auto mat = make_shared<dielectric>(1.5);
    auto material = make_shared<lambertian>(glm::dvec3(0.8, 0.8, 0.0));
    auto material1 = make_shared<lambertian>(glm::dvec3(0.6, 0.8, 0.2));
    objects.add(make_shared<sphere>(glm::dvec3(0,-150,-1), 100, material1));
    objects.add(make_shared<sphere>(glm::dvec3(-3,-1,0), 1.0, material));
    objects.add(make_shared<xy_rect>(0, 2, -5, 2, -1, pertext));
    */
        
    return objects;
}

int main()
{
    std::cout << "Completed Project is running" << std::endl << std::endl;
    
    //dimensions
    //ratio(dimensions.y / dimensions.x)
    //for the camera
    //1920×1080
    //640×480
    PixelBuffer pixel_buffer(glm::ivec2(640, 480));
    
    //position,target,pixel_buffer
    Camera camera(glm::dvec3(0, 5, 5), glm::dvec3(0, 0, 0), pixel_buffer);
    
    hittable_list world;
    world = simple_light();
    
    const int max_depth = 50;
    const int samples_per_pixel = 300;
    
    glm::dvec3 background(0.3,0.5,0.7);
    
    for (int y = 0; y < pixel_buffer.dimensions.y; ++y)
    {
        for (int x = 0; x < pixel_buffer.dimensions.x; ++x)
        {
            
            glm::dvec3 color(0,0,0);
            glm::dvec3 color2(0,0,0);
           
            for (int s = 0; s < samples_per_pixel; ++s)
            {
            
              auto u = x + random_double();
              auto v = y + random_double();
              glm::dvec2 uv(u, v);
              uv /= pixel_buffer.dimensions; //Normalization
             
              //origin, direction
              Ray ray(camera.position, glm::normalize(camera.raster_to_world(uv) - camera.position));
            
              color2 += ray_color(ray, background, world, max_depth);
                
            }
            
            color.x = double(color2.x) / samples_per_pixel;
            color.y = double(color2.y) / samples_per_pixel;
            color.z = double(color2.z) / samples_per_pixel;
            
            //for flipping the image
            //y'nin en ustunden baslar b rengine
            pixel_buffer.set(x, pixel_buffer.dimensions.y - y - 1, color);
                
        }
    
    }
    
    
    std::ofstream output;
    output.open("./render.ppm", std::ios::out | std::ios::trunc);
    if (!output.is_open())
        return 1;

    std::cout << "Outputting...";
    IO::write_as_PPM(pixel_buffer, output, samples_per_pixel);
    output.close();
    std::cout << "done!" << std::endl;

    return 0;
}


